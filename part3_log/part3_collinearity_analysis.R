# ==============================================================================
# DATA 200BP: Homework #2 - Part III: Collinearity Analysis
# Bike Sharing Dataset
# ==============================================================================

# Load required libraries
library(tidyverse)    # Data manipulation and visualization
library(car)          # For VIF calculation
library(corrplot)     # For correlation matrix visualization
library(reshape2)     # For melting correlation matrix

# Set working directory (adjust as needed)
# setwd("~/data210p/hw2")

# ==============================================================================
# 1. LOAD AND PREPARE DATA
# ==============================================================================

# Load the daily bike sharing data
bike_data <- read.csv("day.csv")

# Display structure of the dataset
cat("Dataset Structure:\n")
str(bike_data)
cat("\n")

# Display first few rows
cat("First few rows:\n")
head(bike_data)
cat("\n")

# Summary statistics
cat("Summary Statistics:\n")
summary(bike_data)
cat("\n")

# ==============================================================================
# 2. FIT BASELINE LINEAR MODEL
# ==============================================================================
# Note: You should use the same model specification from Part I
# This is an example model - adjust based on your Part I baseline model

# Select predictors (excluding instant, dteday, casual, registered since cnt = casual + registered)
# Using sensible predictors for bike rentals

# Convert categorical variables to factors
bike_data$season <- as.factor(bike_data$season)
bike_data$yr <- as.factor(bike_data$yr)
bike_data$mnth <- as.factor(bike_data$mnth)
bike_data$holiday <- as.factor(bike_data$holiday)
bike_data$weekday <- as.factor(bike_data$weekday)
bike_data$workingday <- as.factor(bike_data$workingday)
bike_data$weathersit <- as.factor(bike_data$weathersit)

# Fit baseline model with log transformation (from Part II)
# Not including casual and registered as they directly sum to cnt
baseline_model <- lm(log(cnt + 1) ~ season + yr + mnth + holiday + weekday + 
                        weathersit + temp + atemp + 
                       hum + windspeed, 
                     data = bike_data)

cat("Baseline Model Summary:\n")
summary(baseline_model)
cat("\n")

# ==============================================================================
# 3. IDENTIFY NUMERIC PREDICTORS FOR COLLINEARITY ANALYSIS
# ==============================================================================

# Extract numeric predictors used in the model
numeric_predictors <- c("temp", "atemp", "hum", "windspeed")

# If you have transformed variables or interactions, add them here
# For example, if you used log transformation in Part II:
# bike_data$log_cnt <- log(bike_data$cnt + 1)
# Or if you added polynomial terms or interactions

# Create subset with only numeric predictors
numeric_data <- bike_data[, numeric_predictors]

cat("Numeric Predictors for Collinearity Analysis:\n")
print(colnames(numeric_data))
cat("\n")

# ==============================================================================
# 4. COMPUTE CORRELATION MATRIX
# ==============================================================================

# Calculate correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs")

cat("Correlation Matrix:\n")
print(round(cor_matrix, 3))
cat("\n")

# ==============================================================================
# 5. VISUALIZE CORRELATION MATRIX
# ==============================================================================

# Create correlation plot with multiple styles

# Style 1: Color-coded correlation plot with values
png("correlation_matrix_plot1.png", width = 800, height = 800, res = 120)
corrplot(cor_matrix, 
         method = "color",
         type = "upper",
         addCoef.col = "black",
         tl.col = "black",
         tl.srt = 45,
         number.cex = 0.8,
         title = "Correlation Matrix of Numeric Predictors",
         mar = c(0, 0, 2, 0))
dev.off()
cat("Saved: correlation_matrix_plot1.png\n")

# Style 2: Circle plot
png("correlation_matrix_plot2.png", width = 800, height = 800, res = 120)
corrplot(cor_matrix, 
         method = "circle",
         type = "lower",
         addCoef.col = "black",
         tl.col = "black",
         tl.srt = 45,
         number.cex = 0.7,
         title = "Correlation Matrix (Circle Plot)",
         mar = c(0, 0, 2, 0))
dev.off()
cat("Saved: correlation_matrix_plot2.png\n")

# Style 3: Heatmap using ggplot2
cor_melted <- melt(cor_matrix)
ggplot(cor_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), size = 4) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1),
                       name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5)) +
  labs(title = "Correlation Heatmap of Numeric Predictors",
       x = "", y = "")
ggsave("correlation_heatmap.png", width = 8, height = 7, dpi = 120)
cat("Saved: correlation_heatmap.png\n\n")

# ==============================================================================
# 6. IDENTIFY HIGHLY CORRELATED PREDICTOR PAIRS
# ==============================================================================

# Find pairs with correlation > 0.7 (threshold commonly used)
high_cor_threshold <- 0.7

cat("Highly Correlated Predictor Pairs (|r| > ", high_cor_threshold, "):\n", sep = "")
high_cor_pairs <- which(abs(cor_matrix) > high_cor_threshold & 
                          abs(cor_matrix) < 1, arr.ind = TRUE)

if (nrow(high_cor_pairs) > 0) {
  for (i in 1:nrow(high_cor_pairs)) {
    row_idx <- high_cor_pairs[i, 1]
    col_idx <- high_cor_pairs[i, 2]
    if (row_idx < col_idx) {  # Avoid duplicates
      var1 <- rownames(cor_matrix)[row_idx]
      var2 <- colnames(cor_matrix)[col_idx]
      cor_val <- cor_matrix[row_idx, col_idx]
      cat(sprintf("  %s <-> %s: r = %.3f\n", var1, var2, cor_val))
    }
  }
} else {
  cat("  No highly correlated pairs found with threshold ", high_cor_threshold, "\n", sep = "")
}
cat("\n")

# ==============================================================================
# 7. COMPUTE VARIANCE INFLATION FACTORS (VIF)
# ==============================================================================

# Calculate VIF for all predictors in the model
# VIF > 10 indicates severe multicollinearity
# VIF > 5 indicates moderate multicollinearity

cat("Variance Inflation Factors (VIF):\n")
cat("(VIF > 10 = severe, VIF > 5 = moderate multicollinearity)\n\n")

vif_values <- vif(baseline_model)

# For models with categorical variables, vif() returns GVIF
# We'll handle both cases
if (is.matrix(vif_values)) {
  # GVIF for categorical variables
  vif_df <- as.data.frame(vif_values)
  vif_df$Variable <- rownames(vif_df)
  vif_df <- vif_df[, c("Variable", "GVIF", "Df", "GVIF^(1/(2*Df))")]
  print(vif_df, row.names = FALSE)
} else {
  # Regular VIF for continuous variables
  vif_df <- data.frame(
    Variable = names(vif_values),
    VIF = vif_values
  )
  vif_df <- vif_df[order(-vif_df$VIF), ]
  print(vif_df, row.names = FALSE)
  
  # Highlight problematic VIFs
  cat("\nPredictors with VIF > 5:\n")
  high_vif <- vif_df[vif_df$VIF > 5, ]
  if (nrow(high_vif) > 0) {
    print(high_vif, row.names = FALSE)
  } else {
    cat("  None\n")
  }
  
  cat("\nPredictors with VIF > 10:\n")
  severe_vif <- vif_df[vif_df$VIF > 10, ]
  if (nrow(severe_vif) > 0) {
    print(severe_vif, row.names = FALSE)
  } else {
    cat("  None\n")
  }
}
cat("\n")

# ==============================================================================
# 8. VISUALIZE VIF VALUES
# ==============================================================================

# Extract VIF values for numeric predictors
if (is.matrix(vif_values)) {
  # For GVIF, we'll use GVIF^(1/(2*Df)) which is comparable to VIF
  vif_numeric <- vif_values[numeric_predictors, "GVIF^(1/(2*Df))"]
  vif_plot_df <- data.frame(
    Variable = numeric_predictors,
    VIF = vif_numeric
  )
} else {
  vif_plot_df <- data.frame(
    Variable = names(vif_values)[names(vif_values) %in% numeric_predictors],
    VIF = vif_values[names(vif_values) %in% numeric_predictors]
  )
}

# Create VIF bar plot
ggplot(vif_plot_df, aes(x = reorder(Variable, VIF), y = VIF)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_hline(yintercept = 5, color = "orange", linetype = "dashed", size = 1) +
  geom_hline(yintercept = 10, color = "red", linetype = "dashed", size = 1) +
  coord_flip() +
  labs(title = "Variance Inflation Factors (VIF) for Numeric Predictors",
       subtitle = "Dashed lines: Orange = 5 (moderate concern), Red = 10 (severe concern)",
       x = "Predictor",
       y = "VIF") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))
ggsave("vif_plot.png", width = 8, height = 6, dpi = 120)
cat("Saved: vif_plot.png\n\n")

# ==============================================================================
# 9. DETAILED COLLINEARITY ANALYSIS
# ==============================================================================

cat("=== DETAILED COLLINEARITY ANALYSIS ===\n\n")

# Most correlated pair
cor_no_diag <- cor_matrix
diag(cor_no_diag) <- NA
max_cor_idx <- which(abs(cor_no_diag) == max(abs(cor_no_diag), na.rm = TRUE), 
                     arr.ind = TRUE)[1, ]
var1_name <- rownames(cor_matrix)[max_cor_idx[1]]
var2_name <- colnames(cor_matrix)[max_cor_idx[2]]
max_cor_val <- cor_matrix[max_cor_idx[1], max_cor_idx[2]]

cat("Most correlated predictor pair:\n")
cat(sprintf("  %s and %s: r = %.3f\n\n", var1_name, var2_name, max_cor_val))

# Scatterplot of most correlated variables
ggplot(bike_data, aes_string(x = var1_name, y = var2_name)) +
  geom_point(alpha = 0.5, color = "steelblue") +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(title = sprintf("Scatterplot: %s vs %s", var1_name, var2_name),
       subtitle = sprintf("Correlation: r = %.3f", max_cor_val),
       x = var1_name,
       y = var2_name) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))
ggsave("most_correlated_pair.png", width = 8, height = 6, dpi = 120)
cat("Saved: most_correlated_pair.png\n\n")

# ==============================================================================
# 10. STRATEGIES TO ADDRESS COLLINEARITY (EXAMPLES)
# ==============================================================================

cat("=== STRATEGIES TO ADDRESS COLLINEARITY ===\n\n")

cat("Strategy 1: Remove one of the highly correlated predictors\n")
cat(sprintf("  Example: Remove '%s' and keep '%s'\n\n", var2_name, var1_name))

# Fit model without the second variable (with log transformation)
formula_str <- paste("log(cnt + 1) ~ season + yr + mnth + holiday + weekday + weathersit +",
                     paste(numeric_predictors[numeric_predictors != var2_name], collapse = " + "))
model_reduced <- lm(as.formula(formula_str), data = bike_data)

cat("Reduced Model (without", var2_name, "):\n")
print(summary(model_reduced)$coefficients[numeric_predictors[numeric_predictors != var2_name], ])
cat("\n")

# Compare VIF
if (length(numeric_predictors[numeric_predictors != var2_name]) > 1) {
  vif_reduced <- vif(model_reduced)
  cat("VIF after removing", var2_name, ":\n")
  if (is.matrix(vif_reduced)) {
    print(vif_reduced[numeric_predictors[numeric_predictors != var2_name], ])
  } else {
    print(vif_reduced[numeric_predictors[numeric_predictors != var2_name]])
  }
  cat("\n")
}

cat("Strategy 2: Use regularization (Ridge/Lasso - see Part V)\n")
cat("  Ridge regression can handle collinearity by shrinking coefficients\n")
cat("  Lasso can perform variable selection automatically\n\n")

cat("Strategy 3: Principal Components Regression (PCR)\n")
cat("  Transform predictors into uncorrelated principal components\n")
cat("  (Not required for this assignment)\n\n")

# ==============================================================================
# 11. DISCUSSION POINTS FOR REPORT
# ==============================================================================

cat("=== KEY POINTS FOR YOUR REPORT ===\n\n")

cat("1. Correlation Analysis:\n")
cat(sprintf("   - Highest correlation: %s and %s (r = %.3f)\n", 
            var1_name, var2_name, max_cor_val))
cat("   - This indicates [strong/moderate/weak] linear relationship\n\n")

cat("2. VIF Analysis:\n")
if (is.matrix(vif_values)) {
  cat("   - Using GVIF^(1/(2*Df)) for categorical variables\n")
  cat("   - Check for values > 5 (moderate) or > 10 (severe)\n")
} else {
  max_vif_var <- names(which.max(vif_values))
  max_vif_val <- max(vif_values)
  cat(sprintf("   - Highest VIF: %s = %.2f\n", max_vif_var, max_vif_val))
  if (max_vif_val > 10) {
    cat("   - This indicates SEVERE multicollinearity\n")
  } else if (max_vif_val > 5) {
    cat("   - This indicates MODERATE multicollinearity\n")
  } else {
    cat("   - VIF values are acceptable (< 5)\n")
  }
}
cat("\n")

cat("3. Impact on Model:\n")
cat("   - How collinearity affects standard errors: [discuss]\n")
cat("   - Impact on coefficient interpretation: [discuss]\n")
cat("   - Effect on predictive performance: [discuss]\n\n")

cat("4. Mitigation Strategies:\n")
cat("   - Which predictor to remove (if any): [discuss]\n")
cat("   - Justification for keeping/removing: [discuss]\n")
cat("   - Alternative approaches (regularization): [discuss]\n\n")

# ==============================================================================
# 12. ADDITIONAL ANALYSIS: CONDITION NUMBER
# ==============================================================================

cat("=== ADDITIONAL DIAGNOSTIC: CONDITION NUMBER ===\n\n")

# Calculate condition number
# Large condition number (> 30) indicates collinearity
X_matrix <- model.matrix(baseline_model)[, -1]  # Remove intercept
X_numeric <- X_matrix[, numeric_predictors]
condition_number <- kappa(X_numeric, exact = TRUE)

cat(sprintf("Condition Number: %.2f\n", condition_number))
if (condition_number > 30) {
  cat("  Warning: Condition number > 30 indicates potential collinearity issues\n")
} else {
  cat("  Condition number is acceptable (< 30)\n")
}
cat("\n")

# ==============================================================================
# 13. SAVE RESULTS SUMMARY
# ==============================================================================

# Create a summary report
sink("collinearity_analysis_summary.txt")
cat("COLLINEARITY ANALYSIS SUMMARY\n")
cat("==============================\n\n")
cat("Dataset: Bike Sharing (day.csv)\n")
cat("Analysis Date:", format(Sys.Date(), "%Y-%m-%d"), "\n\n")

cat("CORRELATION MATRIX:\n")
print(round(cor_matrix, 3))
cat("\n\n")

cat("VARIANCE INFLATION FACTORS:\n")
if (is.matrix(vif_values)) {
  print(vif_values)
} else {
  print(sort(vif_values, decreasing = TRUE))
}
cat("\n\n")

cat("CONDITION NUMBER:", round(condition_number, 2), "\n\n")

cat("HIGHLY CORRELATED PAIRS (|r| > 0.7):\n")
if (nrow(high_cor_pairs) > 0) {
  for (i in 1:nrow(high_cor_pairs)) {
    row_idx <- high_cor_pairs[i, 1]
    col_idx <- high_cor_pairs[i, 2]
    if (row_idx < col_idx) {
      var1 <- rownames(cor_matrix)[row_idx]
      var2 <- colnames(cor_matrix)[col_idx]
      cor_val <- cor_matrix[row_idx, col_idx]
      cat(sprintf("  %s <-> %s: r = %.3f\n", var1, var2, cor_val))
    }
  }
} else {
  cat("  None found\n")
}

sink()
cat("Saved: collinearity_analysis_summary.txt\n\n")

# ==============================================================================
# END OF SCRIPT
# ==============================================================================

cat("===== PART III COLLINEARITY ANALYSIS COMPLETE =====\n")
cat("\nGenerated files:\n")
cat("  - correlation_matrix_plot1.png\n")
cat("  - correlation_matrix_plot2.png\n")
cat("  - correlation_heatmap.png\n")
cat("  - vif_plot.png\n")
cat("  - most_correlated_pair.png\n")
cat("  - collinearity_analysis_summary.txt\n")
cat("\nAll results have been saved to the working directory.\n")
