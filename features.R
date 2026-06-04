#!/usr/bin/env Rscript

# =========================================================
# Compute ELA features for DTLZ, WFG, and DBMOPP datasets
# =========================================================

# -------------------------
# 1) Custom library path
# -------------------------
.libPaths(c("/projappl/project_2017216/rpackages", .libPaths()))

suppressPackageStartupMessages({
  library(flacco)
  library(dplyr)
})

# -------------------------
# 2) Paths
# -------------------------
root_folder   <- "/scratch/project_2017216/Data"
output_dir    <- "/scratch/project_2017216/modelling_results"
tracking_file <- file.path(output_dir, "processed_files_R.csv")
output_csv    <- file.path(output_dir, "features.csv")

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

# -------------------------
# 3) Folders to process
# -------------------------
folders <- list.dirs(root_folder, recursive = TRUE, full.names = TRUE)
folders <- folders[grepl("DTLZ|WFG|DBMOPP|Engineering", folders, ignore.case = TRUE)]

cat("Found", length(folders), "candidate folders\n")

# -------------------------
# 4) Config
# -------------------------
NORMALIZE_XY <- TRUE
Y_TRANSFORM  <- "rank"   # "rank" | "minmax" | "none"

COLLAPSE_DUP_X     <- TRUE
DEDUP_AGG          <- "median"  # "median" | "mean"
DEDUP_ROUND_DIGITS <- 6

calc_feature_sets <- c(
  "ela_meta", "ela_distr", "basic", "disp",
  "ic", "nbc", "pca", "cm_angle"
)

# -------------------------
# 5) Tracking file
# -------------------------
if (file.exists(tracking_file)) {
  processed_files_df <- read.csv(tracking_file, stringsAsFactors = FALSE)
  needed_cols <- c("file", "f_col")
  for (cc in needed_cols) {
    if (!cc %in% names(processed_files_df)) {
      processed_files_df[[cc]] <- NA_character_
    }
  }
  processed_files_df <- processed_files_df[, needed_cols, drop = FALSE]
} else {
  processed_files_df <- data.frame(
    file = character(),
    f_col = character(),
    stringsAsFactors = FALSE
  )
}

# -------------------------
# 6) Helpers
# -------------------------
clean_column_names <- function(cnames) {
  cnames <- gsub("^x_", "x", cnames)
  cnames <- gsub("^f_", "f", cnames)
  cnames
}

scale01 <- function(v) {
  rng <- range(v, finite = TRUE, na.rm = TRUE)
  if (!all(is.finite(rng)) || diff(rng) == 0) {
    return(rep(0.5, length(v)))
  }
  (v - rng[1]) / (rng[2] - rng[1])
}

rank01 <- function(v) {
  r <- rank(v, ties.method = "average", na.last = "keep")
  rr <- range(r, finite = TRUE, na.rm = TRUE)
  if (!all(is.finite(rr)) || diff(rr) == 0) {
    return(rep(0.5, length(v)))
  }
  (r - rr[1]) / (rr[2] - rr[1])
}

collapse_duplicates <- function(X, y, agg = "median", round_digits = NA) {
  Xw <- as.data.frame(X)

  if (is.numeric(round_digits) && is.finite(round_digits)) {
    Xw[] <- lapply(Xw, function(col) {
      if (is.numeric(col)) round(col, digits = round_digits) else col
    })
  }

  df <- cbind(Xw, .y = y)
  fun <- if (identical(agg, "mean")) mean else median

  out <- stats::aggregate(.y ~ ., data = df, FUN = fun)

  list(
    X = out[, !(names(out) %in% ".y"), drop = FALSE],
    y = out$.y
  )
}

safe_noise_tag <- function(fname) {
  fname_lower <- tolower(fname)

  if (grepl("_noise", fname_lower)) {
    return("noise")
  } else if (grepl("truncnorm|normal", fname_lower)) {
    return("normal")
  } else if (grepl("uniform", fname_lower)) {
    return("uniform")
  } else {
    return("none")
  }
}

safe_feature_set <- function(feat.object, fs_name) {
  tryCatch({
    calculateFeatureSet(feat.object, set = fs_name)
  }, error = function(e) {
    cat("    [WARN] Feature set failed:", fs_name, "|", conditionMessage(e), "\n")
    return(NULL)
  })
}

append_csv_row <- function(df_row, csv_path) {
  if (!file.exists(csv_path)) {
    write.csv(df_row, csv_path, row.names = FALSE)
  } else {
    write.table(
      df_row,
      csv_path,
      sep = ",",
      row.names = FALSE,
      col.names = FALSE,
      append = TRUE
    )
  }
}

already_processed <- function(file, f_col, processed_df) {
  any(processed_df$file == file & processed_df$f_col == f_col)
}

# -------------------------
# 7) Main loop
# -------------------------
for (folder in folders) {
  files <- list.files(folder, full.names = TRUE, pattern = "\\.csv$", ignore.case = TRUE)

  if (length(files) == 0) next

  cat("\n====================================================\n")
  cat("Folder:", folder, "\n")
  cat("Files found:", length(files), "\n")
  cat("====================================================\n")

  for (file in files) {
    cat("\n[INFO] Reading file:", file, "\n")

    dat <- tryCatch(
      read.csv(file, stringsAsFactors = FALSE),
      error = function(e) {
        cat("[WARN] Could not read file:", file, "|", conditionMessage(e), "\n")
        return(NULL)
      }
    )

    if (is.null(dat)) next
    if (nrow(dat) == 0) {
      cat("[WARN] Empty file, skipping:", file, "\n")
      next
    }

    colnames(dat) <- clean_column_names(colnames(dat))

    input_idx  <- grep("^x[0-9]+$", colnames(dat))
    output_idx <- grep("^f[0-9]+$", colnames(dat))

    if (length(input_idx) == 0 || length(output_idx) == 0) {
      cat("[WARN] Could not find x/f columns, skipping:", file, "\n")
      next
    }

    Problem     <- basename(dirname(file))
    VarCount    <- length(input_idx)
    ObjCount    <- length(output_idx)
    num_samples <- nrow(dat)
    noise_tag   <- safe_noise_tag(basename(file))
    is_uniform  <- as.integer(noise_tag == "uniform")

    for (j in output_idx) {
      out_name <- colnames(dat)[j]

      if (already_processed(file, out_name, processed_files_df)) {
        cat("[INFO] Skipping already processed file/objective:", file, "|", out_name, "\n")
        next
      }

      cat("[INFO] Processing objective:", out_name, "\n")

      inputs  <- dat[, input_idx, drop = FALSE]
      outputs <- suppressWarnings(as.numeric(dat[, j]))

      valid_mask <- complete.cases(inputs) & is.finite(outputs)
      inputs  <- inputs[valid_mask, , drop = FALSE]
      outputs <- outputs[valid_mask]

      if (nrow(inputs) < 10) {
        cat("[WARN] Too few valid rows after filtering for", out_name, "- skipping\n")
        next
      }

      before_n <- nrow(inputs)

      if (isTRUE(COLLAPSE_DUP_X)) {
        cd <- collapse_duplicates(
          X = inputs,
          y = outputs,
          agg = DEDUP_AGG,
          round_digits = DEDUP_ROUND_DIGITS
        )
        inputs  <- cd$X
        outputs <- cd$y

        cat(sprintf(
          "  Dedup: %d -> %d rows (agg=%s, round=%s)\n",
          before_n, nrow(inputs), DEDUP_AGG,
          ifelse(is.na(DEDUP_ROUND_DIGITS), "none", as.character(DEDUP_ROUND_DIGITS))
        ))
      }

      if (nrow(inputs) < 10) {
        cat("[WARN] Too few rows after dedup for", out_name, "- skipping\n")
        next
      }

      if (isTRUE(NORMALIZE_XY)) {
        inputs <- as.data.frame(lapply(inputs, scale01))

        if (Y_TRANSFORM == "rank") {
          outputs <- rank01(outputs)
        } else if (Y_TRANSFORM == "minmax") {
          outputs <- scale01(outputs)
        } else if (Y_TRANSFORM == "none") {
          outputs <- outputs
        } else {
          cat("[WARN] Unknown Y_TRANSFORM =", Y_TRANSFORM, "-> using raw y\n")
        }
      }

      feat.object <- tryCatch({
        createFeatureObject(X = inputs, y = outputs)
      }, error = function(e) {
        cat("[WARN] createFeatureObject failed for", file, out_name, "|", conditionMessage(e), "\n")
        return(NULL)
      })

      if (is.null(feat.object)) next

      one_row <- data.frame(
        Problem     = Problem,
        VarCount    = VarCount,
        ObjCount    = ObjCount,
        num_samples = num_samples,
        is_uniform  = is_uniform,
        NoiseTag    = noise_tag,
        f_col       = out_name,
        stringsAsFactors = FALSE
      )

      combined_sets_df <- data.frame(dummy = NA, stringsAsFactors = FALSE)

      for (fs_name in calc_feature_sets) {
        cat("  └─ set:", fs_name, "for", out_name, "\n")

        tmp <- safe_feature_set(feat.object, fs_name)

        if (!is.null(tmp) && length(tmp) > 0) {
          vec <- unlist(tmp)

          if (!all(is.na(vec))) {
            tmp_df <- as.data.frame(t(vec), stringsAsFactors = FALSE)
            colnames(tmp_df) <- paste0(fs_name, "_", names(vec))
            combined_sets_df <- cbind(combined_sets_df, tmp_df)
          } else {
            cat("    [WARN] all NA for", fs_name, "- skipping\n")
          }
        } else {
          cat("    [WARN] empty result for", fs_name, "- skipping\n")
        }
      }

      combined_sets_df <- combined_sets_df[, setdiff(names(combined_sets_df), "dummy"), drop = FALSE]

      if (ncol(combined_sets_df) > 0) {
        out_row <- cbind(one_row, combined_sets_df)
        append_csv_row(out_row, output_csv)
      } else {
        cat("  [WARN] no valid features for", out_name, "- row skipped\n")
      }

      entry <- data.frame(
        file = file,
        f_col = out_name,
        stringsAsFactors = FALSE
      )
      append_csv_row(entry, tracking_file)

      processed_files_df <- rbind(processed_files_df, entry)
    }
  }
}

cat("\n✅ All features written to:", output_csv, "\n")
cat("✅ Tracking file updated at:", tracking_file, "\n")
