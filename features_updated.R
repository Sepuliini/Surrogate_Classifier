# Set the custom library path explicitly in the R script
.libPaths("/projappl/project_2012636/rpackages")

# Load the flacco library
library(flacco)

# Define the root folder where your data is stored
root_folder   = "/scratch/project_2012636/Data"

# Define the output directory and the tracking file
output_dir    = "/scratch/project_2012636/modelling_results"
tracking_file = file.path(output_dir, "processed_files_R.csv")

# List all subfolders within the root directory
folders = list.dirs(root_folder, recursive = TRUE)

# Function to clean column names (handles cases like 'x_1' vs 'x1')
clean_column_names <- function(cnames) {
  gsub("x_", "x", cnames)  # Replaces 'x_' with 'x'
}

# Feature sets from the paper (Mersmann et al. 2011 + additional ones)
# six classical ELA sets + basic, disp, ic, nbc, pca, plus cma (cell mapping angle)
calc_feature_sets = c(
  "ela_conv",         # convexity
  "ela_curv",         # curvature
  "ela_level",        # levelset
  "ela_local",        # local search
  "ela_meta",         # metamodel
  "ela_distribution", # y-distribution
  "basic",            # basic
  "disp",             # dispersion
  "ic",               # information content
  "nbc",              # nearest better clustering
  "pca"               # principal component
)

# Initialize a final data.frame to store one row per (file + f-column)
all_features = data.frame()

# Load existing processed files list, if it exists
if (file.exists(tracking_file)) {
  processed_files <- read.csv(tracking_file, stringsAsFactors = FALSE)$file
} else {
  processed_files <- character()  # Empty if no tracking file exists
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

    # For each output column, calculate the specified feature sets
    for (output_col in output_cols) {
      out_name = colnames(dat)[output_col]
      outputs  = as.numeric(dat[, output_col])

      # Create FeatureObject
      feat.object = createFeatureObject(X = inputs, y = outputs)

      # Prepare one-row data frame to collect all features for this (file + f-col)
      # Start with identifying info
      one_row = data.frame(
        file_name      = file,
        f_col          = out_name,
        num_samples    = num_sample,
        dimensionality = length(input_cols),
        is_uniform     = is_uni,
        stringsAsFactors = FALSE
      )

      # Calculate each feature set in calc_feature_sets
      combined_sets_df = data.frame()
      for (fs_name in calc_feature_sets) {
        cat("  Calculating set:", fs_name, "for", out_name, "\n")
        tmp = data.frame(calculateFeatureSet(feat.object, set = fs_name))
        
        # Rename columns, e.g. "ela_meta_min" -> "ela_meta_min"
        new_names = paste0(fs_name, "_", colnames(tmp))
        colnames(tmp) = new_names

        # Combine columns horizontally
        if (ncol(combined_sets_df) == 0) {
          combined_sets_df = tmp
        } else {
          combined_sets_df = cbind(combined_sets_df, tmp)
        }
      }

      # Now calculate CMA (cell mapping angle) separately
      cat("  Calculating cma features for", out_name, "\n")
      cma_df = data.frame(calculateFeatureSet(
        feat.object,
        set = "cma",
        control = list(cm_blocks = 3, cm_angle = TRUE, cm_conv = FALSE, cm_grad = FALSE)
      ))
      colnames(cma_df) = paste0("cma_", colnames(cma_df))

      # Combine CMA columns
      combined_sets_df = cbind(combined_sets_df, cma_df)

      # Bind the combined features to the identifying columns
      one_row = cbind(one_row, combined_sets_df)

      # Finally, merge into the global data frame
      all_features = rbind(all_features, one_row)
    }

    # Mark this file as processed
    processed_files = c(processed_files, file)
    write.csv(data.frame(file = processed_files), tracking_file, row.names = FALSE)
  }
}

# Write out the final result (one row per file + f-col)
output_csv = file.path(output_dir, "features.csv")
write.csv(all_features, file = output_csv, row.names = FALSE)
cat("\nAll features have been saved to:", output_csv, "\n")
