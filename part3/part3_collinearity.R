setwd("/Users/mishikaahuja/DATA210P/hw2")
# ============================================================
# DATA 200BP: Homework #2 - Part III: Collinearity Analysis
# Dataset: day.csv (Bike Sharing)
# ============================================================

# Load libraries
library(car)
library(corrplot)
library(ggplot2)
library(reshape2)

# ============================================================
# 1. LOAD DATA
# ============================================================
bike <- read.csv("day.csv")

# Convert categorical variables to factors
bike$season     <- as.factor(bike$season)
bike$yr         <- as.factor(bike$yr)
bike$mnth       <- as.factor(bike$mnth)
bike$holiday    <- as.factor(bike$holiday)
bike$weekday    <- as.factor(bike$weekday)
bike$workingday <- as.factor(bike$workingday)
bike$weathersit <- as.factor(bike$weathersit)

# ============================================================
# 2. FIT BASELINE MODEL
# Note: Exclude casual and registered because cnt = casual + registered
# ============================================================
model <- lm(cnt ~ season + yr + mnth + holiday + weekday +
              weathersit + temp + atemp +
              hum + windspeed,
            data = bike)

cat("\n===== BASELINE MODEL SUMMARY =====\n")
summary(model)

# ============================================================
# 3. CORRELATION MATRIX (numeric predictors only)
# ============================================================
numeric_vars <- c("temp", "atemp", "hum", "windspeed")
cor_matrix <- cor(bike[, numeric_vars])

cat("\n===== CORRELATION MATRIX =====\n")
print(round(cor_matrix, 3))

# Plot 1: Correlation heatmap
cor_melted <- melt(cor_matrix)
png("correlation_heatmap.png", width=800, height=700, res=120)
ggplot(cor_melted, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile(color="white") +
  geom_text(aes(label=round(value,2)), size=5) +
  scale_fill_gradient2(low="blue", high="red", mid="white",
                       midpoint=0, limits=c(-1,1), name="r") +
  theme_minimal() +
  theme(axis.text.x=element_text(angle=45, hjust=1),
        plot.title=element_text(hjust=0.5, face="bold")) +
  labs(title="Correlation Matrix - Numeric Predictors", x="", y="")
dev.off()
cat("Saved: correlation_heatmap.png\n")

# Plot 2: corrplot style
png("correlation_corrplot.png", width=700, height=700, res=120)
corrplot(cor_matrix, method="color", type="upper",
         addCoef.col="black", tl.col="black", tl.srt=45,
         number.cex=0.9, mar=c(0,0,2,0),
         title="Correlation Matrix (corrplot)")
dev.off()
cat("Saved: correlation_corrplot.png\n")

# ============================================================
# 4. IDENTIFY HIGHLY CORRELATED PAIRS (|r| > 0.7)
# ============================================================
cat("\n===== HIGHLY CORRELATED PAIRS (|r| > 0.7) =====\n")
for (i in 1:(ncol(cor_matrix)-1)) {
  for (j in (i+1):ncol(cor_matrix)) {
    if (abs(cor_matrix[i,j]) > 0.7) {
      cat(sprintf("  %s <-> %s : r = %.3f\n",
                  rownames(cor_matrix)[i],
                  colnames(cor_matrix)[j],
                  cor_matrix[i,j]))
    }
  }
}

# ============================================================
# 5. VARIANCE INFLATION FACTORS (VIF)
# ============================================================
cat("\n===== VARIANCE INFLATION FACTORS =====\n")
vif_values <- vif(model)
print(vif_values)

# Plot 3: VIF bar chart (numeric predictors only)
if (is.matrix(vif_values)) {
  vif_plot_df <- data.frame(
    Variable = numeric_vars,
    VIF = vif_values[numeric_vars, "GVIF^(1/(2*Df))"]
  )
} else {
  vif_plot_df <- data.frame(
    Variable = numeric_vars,
    VIF = vif_values[numeric_vars]
  )
}

png("vif_plot.png", width=800, height=600, res=120)
ggplot(vif_plot_df, aes(x=reorder(Variable, VIF), y=VIF)) +
  geom_bar(stat="identity", fill="steelblue") +
  geom_hline(yintercept=5,  color="orange", linetype="dashed", linewidth=1.2) +
  geom_hline(yintercept=10, color="red",    linetype="dashed", linewidth=1.2) +
  coord_flip() +
  labs(title="Variance Inflation Factors (VIF)",
       subtitle="Orange = 5 (moderate), Red = 10 (severe)",
       x="Predictor", y="VIF") +
  theme_minimal() +
  theme(plot.title=element_text(hjust=0.5, face="bold"),
        plot.subtitle=element_text(hjust=0.5))
dev.off()
cat("Saved: vif_plot.png\n")

# ============================================================
# 6. SCATTERPLOT OF MOST CORRELATED PAIR
# ============================================================
cor_no_diag <- cor_matrix
diag(cor_no_diag) <- 0
max_idx <- which(abs(cor_no_diag) == max(abs(cor_no_diag)), arr.ind=TRUE)[1,]
var1 <- rownames(cor_matrix)[max_idx[1]]
var2 <- colnames(cor_matrix)[max_idx[2]]
r_val <- cor_matrix[max_idx[1], max_idx[2]]

cat(sprintf("\nMost correlated pair: %s and %s (r = %.3f)\n", var1, var2, r_val))

png("scatterplot_most_correlated.png", width=800, height=600, res=120)
ggplot(bike, aes_string(x=var1, y=var2)) +
  geom_point(alpha=0.5, color="steelblue") +
  geom_smooth(method="lm", color="red", se=TRUE) +
  labs(title=sprintf("Scatterplot: %s vs %s (r = %.3f)", var1, var2, r_val),
       x=var1, y=var2) +
  theme_minimal() +
  theme(plot.title=element_text(hjust=0.5, face="bold"))
dev.off()
cat("Saved: scatterplot_most_correlated.png\n")

# ============================================================
# 7. ADDRESS COLLINEARITY: Remove atemp, refit, compare VIF
# ============================================================
cat("\n===== STRATEGY: REMOVE atemp, REFIT MODEL =====\n")

model_reduced <- lm(cnt ~ season + yr + mnth + holiday + weekday +
                      weathersit + temp +
                      hum + windspeed,
                    data = bike)

cat("\nVIF BEFORE removing atemp:\n")
print(vif(model))
cat("\nVIF AFTER removing atemp:\n")
print(vif(model_reduced))

cat("\n===== DONE. Check your folder for the PNG files. =====\n")
