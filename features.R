library(flacco)

# Define the root folder where your data is stored
root_folder = '/scratch/project_2011092/Data'

# List all subfolders within the root directory
folders = list.dirs(root_folder, recursive = TRUE)

# Initialize variables to store features and additional information
features = NULL
extra_features = data.frame(num_samples = integer(), dimensionality = integer(), is_uniform = integer())
extra_feature_names = c('numsamples', 'dimensionality', 'is_uniform')

# Function to clean column names (to handle cases like 'x_1' vs 'x1')
clean_column_names <- function(colnames) {
  gsub("x_", "x", colnames)  # Replaces 'x_' with 'x'
}

# Loop through all folders and process CSV files within
for (folder in folders) {
  files = list.files(path = folder, full.names = TRUE, pattern = "*.csv")
  
  for (file in files) {
    dat = read.csv(file)
    
    # Clean column names to handle cases like 'x_1' and 'x1'
    colnames(dat) <- clean_column_names(colnames(dat))
    
    num_sample = nrow(dat)
    num_cols = ncol(dat)
    is_uni = grepl('uniform', file)*1  # Identify if the dataset might be from a uniform distribution
    
    # Identify the input columns (those starting with 'x') and output columns (those starting with 'f')
    input_cols = grep("^x", colnames(dat))
    output_cols = grep("^f", colnames(dat))
    
    inputs = dat[, input_cols]
    inputs = apply(inputs, 2, as.numeric)
    
    # Loop through each output column and calculate meta-features for each one
    for (output_col in output_cols) {
      outputs = dat[, output_col]
      outputs = as.numeric(outputs)
      
      feat.object = createFeatureObject(X = inputs, y = outputs)
      
      # Calculate ELA meta-features
      feature_set = data.frame(calculateFeatureSet(feat.object, set = "ela_meta"))
      
      # Create extra feature info (number of samples, dimensionality, uniformity flag)
      new_extra = data.frame(num_sample, length(input_cols), is_uni)
      names(new_extra) = extra_feature_names
      extra_features = rbind(extra_features, new_extra)
      
      # Combine the features and extra info for each file/output
      features = rbind(features, feature_set)
    }
  }
}

# Combine all features with filenames and extra feature information
files_df = data.frame(files = list.files(path = root_folder, recursive = TRUE, full.names = TRUE))
features = cbind(files_df, features, extra_features)

# Write the results to a CSV file
write.csv(features, file = '/scratch/project_2011092/features.csv')
