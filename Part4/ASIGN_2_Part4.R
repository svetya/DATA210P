# Set working directory 
setwd("C:/Users/nisar/OneDrive/Desktop/Quarter/WINTER/SM/SM_ASM_2/bike/bike+sharing+dataset")

# ============================================================
# DATA 200BP: Homework #2 - Part IV: Model Selection & Validation
# Dataset: day.csv (Bike Sharing)
# Response: log(cnt + 1) — transformed for better linear model assumptions
# ============================================================

# Load libraries
library(ggplot2)
library(car)          # for vif()
library(boot)         # for cv.glm()
library(MASS)         # for stepwise selection via AIC
library(reshape2)     # for melt 
library(scales)       # for percent_format()

set.seed(210)

# ============================================================
# 1. LOAD & CLEAN DATA  (mirrors teammate's Part III setup)
# ============================================================
bike <- read.csv("day.csv")

bike$season     <- as.factor(bike$season)
bike$yr         <- as.factor(bike$yr)
bike$mnth       <- as.factor(bike$mnth)
bike$holiday    <- as.factor(bike$holiday)
bike$weekday    <- as.factor(bike$weekday)
bike$workingday <- as.factor(bike$workingday)
bike$weathersit <- as.factor(bike$weathersit)

# Remove instant, dteday (row ID / date string)
bike <- bike[, !names(bike) %in% c("instant", "dteday")]

# Remove leakage: casual + registered = cnt
bike_model <- bike[, !names(bike) %in% c("casual", "registered")]

# ============================================================
# 2. CREATE TRANSFORMED RESPONSE VARIABLE
#    log(cnt + 1) to stabilize variance & improve normality
# ============================================================
bike_model$log_cnt <- log(bike_model$cnt + 1)

cat("\n===== TRANSFORMATION APPLIED =====\n")
cat("  Original response: cnt\n")
cat("  Transformed response: log(cnt + 1)\n")
cat(sprintf("  Range of cnt       : [%d, %d]\n", 
            min(bike_model$cnt), max(bike_model$cnt)))
cat(sprintf("  Range of log_cnt   : [%.3f, %.3f]\n", 
            min(bike_model$log_cnt), max(bike_model$log_cnt)))

# ============================================================
# 3. BASELINE MODEL (using log_cnt as response)
# ============================================================
baseline <- lm(log_cnt ~ season + yr + mnth + holiday + weekday +
                 weathersit + temp + hum + windspeed,
               data = bike_model)

cat("\n===== BASELINE MODEL SUMMARY (log scale) =====\n")
summary(baseline)

# ============================================================
# 4. K-FOLD CROSS-VALIDATION (K = 10) — BASELINE
#    IMPORTANT: We'll compute RMSE on ORIGINAL SCALE (cnt)
#    by back-transforming predictions: exp(pred) - 1
# ============================================================
cat("\n===== 10-FOLD CV: BASELINE MODEL =====\n")

n <- nrow(bike_model)
folds <- sample(rep(1:10, length.out = n))

# Storage for fold-level metrics
fold_rmse_log <- numeric(10)      # RMSE on log scale
fold_rmse_original <- numeric(10) # RMSE on original scale (back-transformed)

for (k in 1:10) {
  test_idx  <- which(folds == k)
  train_idx <- which(folds != k)
  
  # Fit on training fold
  fit_k <- lm(log_cnt ~ season + yr + mnth + holiday + weekday +
                weathersit + temp + hum + windspeed,
              data = bike_model[train_idx, ])
  
  # Predict on test fold (log scale)
  pred_log <- predict(fit_k, newdata = bike_model[test_idx, ])
  
  # Back-transform to original scale: exp(log_cnt) - 1 = cnt
  pred_original <- exp(pred_log) - 1
  
  # Actual values
  actual_log <- bike_model$log_cnt[test_idx]
  actual_original <- bike_model$cnt[test_idx]
  
  # RMSE on log scale
  fold_rmse_log[k] <- sqrt(mean((actual_log - pred_log)^2))
  
  # RMSE on original scale 
  fold_rmse_original[k] <- sqrt(mean((actual_original - pred_original)^2))
}

cat("\n  Per-fold RMSE (LOG scale):\n")
print(round(fold_rmse_log, 4))
cat(sprintf("  Mean RMSE (log)  : %.4f\n", mean(fold_rmse_log)))
cat(sprintf("  SD   RMSE (log)  : %.4f\n", sd(fold_rmse_log)))

cat("\n  Per-fold RMSE (ORIGINAL scale):\n")
print(round(fold_rmse_original, 2))
cat(sprintf("  Mean RMSE (orig) : %.2f rentals\n", mean(fold_rmse_original)))
cat(sprintf("  SD   RMSE (orig) : %.2f rentals\n", sd(fold_rmse_original)))

# Plot 1: Fold-level RMSE — Baseline (Original Scale)
png("cv_baseline_fold_rmse.png", width = 800, height = 500, res = 120)
ggplot(data.frame(Fold = 1:10, RMSE = fold_rmse_original),
       aes(x = Fold, y = RMSE)) +
  geom_bar(stat = "identity", fill = "steelblue", color = "white") +
  geom_hline(yintercept = mean(fold_rmse_original), color = "red",
             linetype = "dashed", linewidth = 1.2) +
  scale_x_continuous(breaks = 1:10) +
  labs(title    = "10-Fold CV: Per-Fold RMSE (Baseline Model)",
       subtitle = sprintf("Mean RMSE = %.2f  |  SD = %.2f  [Original Scale]",
                          mean(fold_rmse_original), sd(fold_rmse_original)),
       x = "Fold", y = "RMSE (rentals)") +
  theme_minimal() +
  theme(plot.title    = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))
dev.off()
cat("\nSaved: cv_baseline_fold_rmse.png\n")

# ============================================================
# 5. STEPWISE MODEL SELECTION  (criterion = AIC)
#    Strategy: start from the full baseline, do backward
#    elimination, then forward selection from the null —
#    report whichever gives the lowest AIC.
# ============================================================
cat("\n===== STEPWISE MODEL SELECTION (AIC) =====\n")

# Full model (baseline) and null model for stepwise bounds
null_model <- lm(log_cnt ~ 1, data = bike_model)

# --- Backward elimination ---
cat("\n--- Backward Elimination ---\n")
step_back <- step(baseline, direction = "backward", trace = 0)
cat("Backward-selected model formula:\n")
print(formula(step_back))
cat(sprintf("AIC (backward) : %.2f\n", AIC(step_back)))

# --- Forward selection ---
cat("\n--- Forward Selection ---\n")
step_fwd <- step(null_model,
                 scope = list(lower = null_model, upper = baseline),
                 direction = "forward", trace = 0)
cat("Forward-selected model formula:\n")
print(formula(step_fwd))
cat(sprintf("AIC (forward)  : %.2f\n", AIC(step_fwd)))

# --- Stepwise (both directions) ---
cat("\n--- Stepwise (Both Directions) ---\n")
step_both <- step(baseline,
                  scope = list(lower = null_model, upper = baseline),
                  direction = "both", trace = 0)
cat("Stepwise-selected model formula:\n")
print(formula(step_both))
cat(sprintf("AIC (stepwise) : %.2f\n", AIC(step_both)))

# Pick the selected model with the lowest AIC
aic_vec <- c(backward  = AIC(step_back),
             forward   = AIC(step_fwd),
             stepwise  = AIC(step_both))
best_dir <- names(which.min(aic_vec))
cat(sprintf("\nBest direction: %s  (AIC = %.2f)\n", best_dir, min(aic_vec)))

# Assign the winning model to 'selected'
selected <- switch(best_dir,
                   backward  = step_back,
                   forward   = step_fwd,
                   stepwise  = step_both)

cat("\n===== SELECTED MODEL SUMMARY (log scale) =====\n")
summary(selected)

# ============================================================
# 6. K-FOLD CV ON THE SELECTED MODEL (same folds)
# ============================================================
cat("\n===== 10-FOLD CV: SELECTED MODEL =====\n")

selected_formula <- formula(selected)

fold_rmse_sel_log <- numeric(10)
fold_rmse_sel_original <- numeric(10)

for (k in 1:10) {
  test_idx  <- which(folds == k)
  train_idx <- which(folds != k)
  
  fit_k <- lm(selected_formula, data = bike_model[train_idx, ])
  
  pred_log <- predict(fit_k, newdata = bike_model[test_idx, ])
  pred_original <- exp(pred_log) - 1
  
  actual_log <- bike_model$log_cnt[test_idx]
  actual_original <- bike_model$cnt[test_idx]
  
  fold_rmse_sel_log[k] <- sqrt(mean((actual_log - pred_log)^2))
  fold_rmse_sel_original[k] <- sqrt(mean((actual_original - pred_original)^2))
}

cat("\n  Per-fold RMSE (LOG scale):\n")
print(round(fold_rmse_sel_log, 4))
cat(sprintf("  Mean RMSE (log)  : %.4f\n", mean(fold_rmse_sel_log)))
cat(sprintf("  SD   RMSE (log)  : %.4f\n", sd(fold_rmse_sel_log)))

cat("\n  Per-fold RMSE (ORIGINAL scale):\n")
print(round(fold_rmse_sel_original, 2))
cat(sprintf("  Mean RMSE (orig) : %.2f rentals\n", mean(fold_rmse_sel_original)))
cat(sprintf("  SD   RMSE (orig) : %.2f rentals\n", sd(fold_rmse_sel_original)))

# ============================================================
# 7. COMPARISON PLOTS
# ============================================================

# --- Plot 2: Side-by-side fold RMSE comparison (Original Scale) ---
compare_df <- data.frame(
  Fold     = rep(1:10, 2),
  RMSE     = c(fold_rmse_original, fold_rmse_sel_original),
  Model    = rep(c("Baseline", "Selected"), each = 10)
)

png("cv_comparison_fold_rmse.png", width = 850, height = 520, res = 120)
ggplot(compare_df, aes(x = Fold, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", color = "white") +
  scale_fill_manual(values = c(Baseline = "steelblue", Selected = "#E8855A")) +
  scale_x_continuous(breaks = 1:10) +
  geom_hline(yintercept = mean(fold_rmse_original), color = "steelblue",
             linetype = "dashed", linewidth = 1.1) +
  geom_hline(yintercept = mean(fold_rmse_sel_original), color = "#E8855A",
             linetype = "dashed", linewidth = 1.1) +
  labs(title    = "10-Fold CV: Baseline vs Selected Model",
       subtitle = sprintf("Mean RMSE — Baseline: %.2f  |  Selected: %.2f  [Original Scale]",
                          mean(fold_rmse_original), mean(fold_rmse_sel_original)),
       x = "Fold", y = "RMSE (rentals)", fill = "Model") +
  theme_minimal() +
  theme(plot.title    = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5))
dev.off()
cat("Saved: cv_comparison_fold_rmse.png\n")

# --- Plot 3: Summary bar chart (in-sample R² + CV RMSE) ---
summary_baseline  <- summary(baseline)
summary_selected  <- summary(selected)

metrics_df <- data.frame(
  Model       = c("Baseline", "Selected"),
  R2          = c(summary_baseline$r.squared,  summary_selected$r.squared),
  Adj_R2      = c(summary_baseline$adj.r.squared, summary_selected$adj.r.squared),
  CV_RMSE_log = c(mean(fold_rmse_log), mean(fold_rmse_sel_log)),
  CV_RMSE_orig = c(mean(fold_rmse_original), mean(fold_rmse_sel_original)),
  n_params    = c(length(coef(baseline)), length(coef(selected)))
)

cat("\n===== COMPARISON TABLE =====\n")
print(metrics_df)

# Plot 3a: R² comparison (these are on log scale)
png("comparison_R2.png", width = 700, height = 450, res = 120)
ggplot(metrics_df, aes(x = Model, y = Adj_R2, fill = Model)) +
  geom_bar(stat = "identity", color = "white", width = 0.5) +
  scale_fill_manual(values = c(Baseline = "steelblue", Selected = "#E8855A")) +
  scale_y_continuous(limits = c(0, 1), labels = percent_format()) +
  geom_text(aes(label = sprintf("%.3f", Adj_R2)), vjust = -0.5, size = 4.5) +
  labs(title = "Adjusted R² — Baseline vs Selected Model",
       subtitle = "[Measured on log(cnt+1) scale]",
       y     = "Adjusted R²") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "none")
dev.off()
cat("Saved: comparison_R2.png\n")

# Plot 3b: CV RMSE comparison (Original Scale)
png("comparison_CV_RMSE.png", width = 700, height = 650, res = 120)
ggplot(metrics_df, aes(x = Model, y = CV_RMSE_orig, fill = Model)) +
  geom_bar(stat = "identity", color = "white", width = 0.5) +
  scale_fill_manual(values = c(Baseline = "steelblue", Selected = "#E8855A")) +
  geom_text(aes(label = sprintf("%.1f", CV_RMSE_orig)), vjust = -0.5, size = 4.5) +
  labs(title = "CV RMSE — Baseline vs Selected Model",
       subtitle = "[Measured on original scale: rentals]",
       y     = "Mean 10-Fold RMSE (rentals)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "none")
dev.off()
cat("Saved: comparison_CV_RMSE.png\n")

# ============================================================
# 8. RESIDUAL DIAGNOSTICS ON SELECTED MODEL (log scale)
# ============================================================
cat("\n===== RESIDUAL DIAGNOSTICS (Selected Model, Log Scale) =====\n")

png("selected_diagnostics.png", width = 900, height = 700, res = 120)
par(mfrow = c(2, 2))
plot(selected, which = 1:4)
par(mfrow = c(1, 1))
dev.off()
cat("Saved: selected_diagnostics.png\n")

# ============================================================
# 9. EXAMPLE PREDICTIONS: Compare log vs original scale
# ============================================================
cat("\n===== EXAMPLE PREDICTIONS (First 5 Observations) =====\n")

pred_log_example <- predict(selected, newdata = bike_model[1:5, ])
pred_original_example <- exp(pred_log_example) - 1

example_df <- data.frame(
  Obs = 1:5,
  Actual_cnt = bike_model$cnt[1:5],
  Pred_log = round(pred_log_example, 3),
  Pred_cnt = round(pred_original_example, 1)
)
print(example_df)

# ============================================================
# 10. FINAL SUMMARY OUTPUT
# ============================================================
cat("\n============================================================\n")
cat("  PART IV — FINAL SUMMARY\n")
cat("============================================================\n")
cat("  Response variable    : log(cnt + 1)\n")
cat("  Evaluation metric    : RMSE on ORIGINAL scale (rentals)\n")
cat(sprintf("  Baseline predictors  : %d (including intercept)\n", length(coef(baseline))))
cat(sprintf("  Selected predictors  : %d (including intercept)\n", length(coef(selected))))
cat(sprintf("  Selection method     : Stepwise (AIC), best = %s\n", best_dir))
cat("  ---\n")
cat("  IN-SAMPLE METRICS (log scale):\n")
cat(sprintf("    Baseline  Adj R²     : %.4f\n", summary_baseline$adj.r.squared))
cat(sprintf("    Selected  Adj R²     : %.4f\n", summary_selected$adj.r.squared))
cat("  ---\n")
cat("  CROSS-VALIDATION (original scale):\n")
cat(sprintf("    Baseline  CV RMSE    : %.2f rentals\n", mean(fold_rmse_original)))
cat(sprintf("    Selected  CV RMSE    : %.2f rentals\n", mean(fold_rmse_sel_original)))
cat(sprintf("    Improvement          : %.2f rentals (%.1f%%)\n", 
            mean(fold_rmse_original) - mean(fold_rmse_sel_original),
            100 * (1 - mean(fold_rmse_sel_original) / mean(fold_rmse_original))))
cat("============================================================\n")
cat("  Saved plots:\n")
cat("    1. cv_baseline_fold_rmse.png\n")
cat("    2. cv_comparison_fold_rmse.png\n")
cat("    3. comparison_R2.png\n")
cat("    4. comparison_CV_RMSE.png\n")
cat("    5. selected_diagnostics.png\n")
cat("============================================================\n")
