# Set the custom library path explicitly in the R script
.libPaths("/projappl/project_2012636/rpackages")

# Load the flacco library
library(flacco)

# Define the root folder where your data is stored
root_folder   = "/scratch/project_2012636/Data"

# Define the output directory and the tracking file
output_dir    = "/scratch/project_2012636/modelling_results"
tracking_file = file.path(output_dir, "processed_files_R.csv")
output_csv = file.path(output_dir, "features.csv")

# List all subfolders within the root directory
folders = list.dirs(root_folder, recursive = TRUE)

# Function to clean column names (handles cases like 'x_1' vs 'x1')
clean_column_names <- function(cnames) {
  gsub("x_", "x", cnames)  # Replaces 'x_' with 'x'
}

# Corrected feature sets
calc_feature_sets = c(
  # "ela_conv",         # convexity (requires an exact function)
  # "ela_curv",         # curvature (requires an exact function)
  # "ela_level",        # levelset (requires an exact function)
  # "ela_local",        # local search (requires an exact function)
  "ela_meta",         # metamodel features
  "ela_distr",        # y-distribution
  "basic",            # basic features
  "disp",             # dispersion features
  "ic",               # information content features
  "nbc",              # nearest better clustering
  "pca",              # principal component analysis features
  "cm_angle"          # corrected from "cma" to "cm_angle" (cell mapping angle features)
)

# Initialize a final data.frame to store one row per (file + f-column)
all_features = data.frame()

# Load existing processed files list, considering skipped features
if (file.exists(tracking_file)) {
  processed_files_df <- read.csv(tracking_file, stringsAsFactors = FALSE)
  processed_files <- processed_files_df$file
} else {
  processed_files <- character()  # Empty if no tracking file exists
}

# Clear the output CSV to avoid old data
if (file.exists(output_csv)) {
  file.remove(output_csv)
}

# Loop over subfolders
for (folder in folders) {
  # Get all .csv files
  files = list.files(path = folder, full.names = TRUE, pattern = "\\.csv$")

  for (file in files) {
    # Skip the file if it's already processed
    if (file %in% processed_files) {
      cat("Skipping already processed file:", file, "\n")
      next
    }

    cat("\nProcessing file:", file, "\n")
    dat = read.csv(file)
    colnames(dat) = clean_column_names(colnames(dat))

    num_sample = nrow(dat)
    num_cols   = ncol(dat)
    # Check if the file name contains 'uniform'
    is_uni = as.integer(grepl("uniform", file)) 

    # Identify the input columns (starting with 'x') and output columns (starting with 'f')
    input_cols  = grep("^x", colnames(dat))
    output_cols = grep("^f", colnames(dat))

    # Convert inputs to numeric
    inputs = dat[, input_cols, drop = FALSE]
    inputs = apply(inputs, 2, as.numeric)

    # Keep track of skipped features
    skipped_features = c()

    # For each output column, calculate the specified feature sets
    for (output_col in output_cols) {
      out_name = colnames(dat)[output_col]
      outputs  = as.numeric(dat[, output_col])

      # Create FeatureObject
      feat.object = createFeatureObject(X = inputs, y = outputs)

      # Prepare one-row data frame to collect all features for this (file + f-col)
      one_row = data.frame(
        file_name      = file,
        f_col          = out_name,
        num_samples    = num_sample,
        dimensionality = length(input_cols),
        is_uniform     = is_uni,
        stringsAsFactors = FALSE
      )

      # Calculate each feature set and check for empty results
      combined_sets_df = data.frame()
      for (fs_name in calc_feature_sets) {
        cat("  Calculating set:", fs_name, "for", out_name, "\n")
        tmp = tryCatch({
          data.frame(calculateFeatureSet(feat.object, set = fs_name))
        }, error = function(e) {
          cat("  Warning: No features generated for", fs_name, "\n")
          skipped_features = c(skipped_features, fs_name)
          return(data.frame())
        })

        # Only add features if non-empty and matching row counts
        if (nrow(tmp) > 0 && ncol(tmp) > 0 && nrow(tmp) == nrow(one_row)) {
          colnames(tmp) = paste0(fs_name, "_", colnames(tmp))
          combined_sets_df = cbind(combined_sets_df, tmp)
        } else {
          cat("  Skipping feature set due to row count mismatch or empty results: ", fs_name, "\n")
        }
      }

      # Bind the combined features to the identifying columns if data exists
      if (ncol(combined_sets_df) > 0) {
        one_row = cbind(one_row, combined_sets_df)

        # Save results incrementally after each file is processed
        if (!file.exists(output_csv)) {
          write.csv(one_row, file = output_csv, row.names = FALSE)
        } else {
          write.table(one_row, file = output_csv, sep = ",", col.names = FALSE, row.names = FALSE, append = TRUE)
        }
      } else {
        cat("  No valid features generated for", out_name, "\n")
      }

      # Prepare processed file entry including skipped features
      processed_entry <- data.frame(file = file, skipped_features = paste(skipped_features, collapse = ", "))

      # Log the processed file and skipped features to the tracking file
      if (!file.exists(tracking_file)) {
        write.csv(processed_entry, tracking_file, row.names = FALSE)
      } else {
        write.table(processed_entry, file = tracking_file, sep = ",", col.names = FALSE, row.names = FALSE, append = TRUE)
      }
    }
  }
}

cat("\nAll features have been saved to:", output_csv, "\n")
