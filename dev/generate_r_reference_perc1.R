#!/usr/bin/env Rscript
# Generate R reference results for Python parity testing with perc=1.0.
# This is the same as generate_r_reference.R but uses perc=1 (GSimp's actual
# default) instead of perc=0.01. Output directories have a _perc1 suffix.
# Skips synthetic_5000 and synthetic_20000 (slow).

library(MDImputeKnnTn)

ref_dir <- "/Users/giuseppeinfusini/wd/md-repos/impute-knn-tn/tests/reference"
gsimp_dir <- file.path(getwd(), "dev", "gsimp_data")

# Helper: run imputeKNN on a matrix and save all reference outputs
# log2: if TRUE, log2-transform before imputation and 2^x after (matching kNN_TN pipeline)
run_and_save <- function(mat, name, k = 5, perc = 1, log2 = FALSE) {
  for (dist_mode in c("truncation", "correlation")) {
    out_dir <- file.path(ref_dir, paste0(name, "_", dist_mode, "_perc1"))
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

    # Save input matrix (raw scale)
    write.csv(mat, file.path(out_dir, "input_matrix.csv"))

    # Work in log2 space if requested (matches kNN_TN pipeline)
    work_mat <- if (log2) log2(mat) else mat

    # Get ParamEstim (only meaningful for truncation)
    param_est <- MDImputeKnnTn:::EstimatesComputation(work_mat, perc = perc)
    write.csv(param_est, file.path(out_dir, "param_estimates.csv"))

    # Run imputation and time it
    t1 <- proc.time()
    result <- MDImputeKnnTn:::imputeKNN(work_mat, k = k, distance = dist_mode, perc = perc)
    t2 <- proc.time()
    elapsed <- (t2 - t1)["elapsed"]

    # Back-transform if log2
    if (log2) result <- 2^result

    write.csv(result, file.path(out_dir, "output_matrix.csv"))
    writeLines(sprintf("%.6f", elapsed), file.path(out_dir, "timing.txt"))
    writeLines(ifelse(log2, "true", "false"), file.path(out_dir, "log2.txt"))

    cat(sprintf("  %s/%s_perc1: %.3fs, NAs remaining: %d\n",
                name, dist_mode, elapsed, sum(is.na(result))))
  }
}

# ============================================================
# 1. GSimp datasets with real missingness
# ============================================================
cat("\n=== GSimp datasets (real missingness, perc=1) ===\n")

for (f in c("targeted_data.csv", "untargeted_data.csv")) {
  d <- read.csv(file.path(gsimp_dir, f), row.names = 1)
  mat <- t(as.matrix(d))  # transpose: features x samples
  name <- sub("_data\\.csv$", "", f)
  cat(sprintf("%s: %d features x %d samples, %d NAs\n",
              name, nrow(mat), ncol(mat), sum(is.na(mat))))
  # Intensity data: use log2 transform (values are large)
  run_and_save(mat, name, log2 = TRUE)
}

# ============================================================
# 2. GSimp datasets needing missingness injection
# ============================================================
cat("\n=== GSimp datasets (inject missingness, perc=1) ===\n")

inject_mnar_missingness <- function(mat, frac = 0.05, seed = 42) {
  # Inject MNAR-style missingness: lower values more likely to be missing
  set.seed(seed)
  for (i in 1:nrow(mat)) {
    row_vals <- mat[i, ]
    n_miss <- max(1, round(length(row_vals) * frac))
    # Probability of being missing is inversely proportional to value rank
    ranks <- rank(row_vals, ties.method = "random")
    probs <- (max(ranks) - ranks + 1) / sum(max(ranks) - ranks + 1)
    miss_idx <- sample(1:length(row_vals), size = n_miss, prob = probs)
    mat[i, miss_idx] <- NA
  }
  return(mat)
}

for (f in c("real_data.csv", "data_sim.csv")) {
  d <- read.csv(file.path(gsimp_dir, f), row.names = 1)
  mat <- t(as.matrix(d))
  mat <- inject_mnar_missingness(mat, frac = 0.05)
  name <- sub("_data\\.csv$", "", sub("\\.csv$", "", f))
  if (name == "data_sim") name <- "sim"
  cat(sprintf("%s: %d features x %d samples, %d NAs\n",
              name, nrow(mat), ncol(mat), sum(is.na(mat))))
  # real_data has intensity values (use log2), sim has small values (no log2)
  use_log2 <- (f == "real_data.csv")
  run_and_save(mat, name, log2 = use_log2)
}

# ============================================================
# 3. bojkova2020 (proteomics, via the full pipeline)
# ============================================================
cat("\n=== bojkova2020 (perc=1) ===\n")
data(bojkova2020)

for (dist_mode in c("truncation", "correlation")) {
  out_dir <- file.path(ref_dir, paste0("bojkova2020_", dist_mode, "_perc1"))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  int <- bojkova2020$protein_intensity
  meta <- bojkova2020$protein_metadata

  write.csv(int, file.path(out_dir, "input_intensities.csv"), row.names = FALSE)
  write.csv(meta, file.path(out_dir, "input_metadata.csv"), row.names = FALSE)

  t1 <- proc.time()
  result <- imputekNN_tn(int, meta, k = 5, distance = dist_mode, perc = 1)
  t2 <- proc.time()
  elapsed <- (t2 - t1)["elapsed"]

  write.csv(result$intensity, file.path(out_dir, "output_intensities.csv"), row.names = FALSE)
  write.csv(result$intensity[result$intensity$Imputed == 1, ],
            file.path(out_dir, "imputed_rows.csv"), row.names = FALSE)
  write.csv(result$runtimeMetadata, file.path(out_dir, "runtime_metadata.csv"), row.names = FALSE)
  writeLines(sprintf("%.6f", elapsed), file.path(out_dir, "timing.txt"))

  cat(sprintf("  bojkova2020/%s_perc1: %.3fs, NAs: %d\n",
              dist_mode, elapsed, sum(is.na(result$intensity$NormalisedIntensity))))
}

# ============================================================
# 4. Synthetic dataset (1K features only, skip 5K and 20K)
# ============================================================
cat("\n=== Synthetic dataset (perc=1) ===\n")

generate_synthetic <- function(n_features, n_reps = 50, seed = 42) {
  set.seed(seed)
  mat <- matrix(NA, nrow = n_features, ncol = n_reps)

  # 70% high abundance
  n_high <- round(n_features * 0.7)
  for (i in 1:n_high) {
    mat[i, ] <- rnorm(n_reps, mean = runif(1, 8, 14), sd = runif(1, 0.3, 1.0))
  }

  # 30% low abundance (near LOD, triggers N-R)
  for (i in (n_high + 1):n_features) {
    mat[i, ] <- rnorm(n_reps, mean = runif(1, 0.5, 2.0), sd = runif(1, 1.0, 2.0))
  }

  # Inject sparse missingness (~2% of cells)
  n_miss <- round(length(mat) * 0.02)
  miss_idx <- sample(length(mat), n_miss)
  mat[miss_idx] <- NA

  colnames(mat) <- paste0("sample_", 1:n_reps)
  rownames(mat) <- paste0("feature_", 1:n_features)
  return(mat)
}

for (n_feat in c(1000)) {
  name <- paste0("synthetic_", n_feat)
  mat <- generate_synthetic(n_feat)

  # Check N-R triggering
  nsamp <- ncol(mat)
  na.sum <- apply(mat, 1, function(x) sum(is.na(x)))
  means <- rowMeans(mat, na.rm = TRUE)
  sds <- apply(mat, 1, function(x) sd(x, na.rm = TRUE))
  lod <- min(mat, na.rm = TRUE)
  idx1 <- which(na.sum / nsamp >= 0.01)
  idx2 <- which(means > 3 * sds + lod)
  idx.nr <- setdiff(1:nrow(mat), c(idx1, idx2))

  cat(sprintf("%s: %d features x %d samples, %d NAs, %d rows trigger N-R\n",
              name, nrow(mat), ncol(mat), sum(is.na(mat)), length(idx.nr)))

  run_and_save(mat, name)
}

cat("\n=== Done ===\n")
cat("Reference files saved to:", ref_dir, "\n")
