
library(tidyverse)
library(caret)
library(randomForest)
library(reshape2)
library(ggplot2)

data <- read.csv("~/Desktop/BIOT670i/airborne_microbes.csv", stringsAsFactors = FALSE)

cat("Columns in dataset:\n")
print(names(data))


names(data) <- make.names(names(data))

# numeric values
num_vars <- c(
  "Temp..In.C",
  "Wind.Speed..MPH.",
  "Relative.Humidity....",
  "Altitude..km.above.sea.level.",
  "SO4.Concentration.ug.m.3",
  "Organic.Carbon.Concentration.ug.m.3",
  "Dust.Concentration.ug.m.3",
  "Black.Carbon.Concentration.ug.m.3"
)

ml_data <- data %>%
  select(Concentration..cfu.m.3., all_of(num_vars)) %>%
  drop_na()

cat("\nPreview of ML-ready data:\n")
print(head(ml_data))

set.seed(123)
trainIndex <- createDataPartition(ml_data$Concentration..cfu.m.3., p = 0.8, list = FALSE)
train <- ml_data[trainIndex, ]
test  <- ml_data[-trainIndex, ]

# random forest model R
rf_model <- randomForest(Concentration..cfu.m.3. ~ ., data = train, importance = TRUE, ntree = 500)
cat("\nRandom Forest Model Summary:\n")
print(rf_model)

# prediction method
pred <- predict(rf_model, test)
R2 <- cor(test$Concentration..cfu.m.3., pred)^2
cat("\nModel Performance (R²): ", round(R2, 3), "\n")

# --- Feature Importance Plot ---
imp_plot <- ggplot(data.frame(Variable = rownames(importance(rf_model)), 
                              Importance = importance(rf_model)[, 1]),
                   aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  theme_minimal(base_size = 14) +
  labs(title = "Random Forest Prediction", 
       x = "Environmental Feature", 
       y = "Importance")

print(imp_plot)
ggsave("variable_importance_plot.png", imp_plot, width = 8, height = 6, dpi = 300)

# --- Observed vs Predicted Plot ---
obs_pred_plot <- ggplot(data.frame(Observed = test$Concentration..cfu.m.3., Predicted = pred),
                        aes(x = Observed, y = Predicted)) +
  geom_point(color = "darkred", alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  theme_minimal(base_size = 14) +
  labs(title = "Observed vs Predicted Microbial Concentration",
       x = "Observed (cfu/m³)", 
       y = "Predicted (cfu/m³)")

print(obs_pred_plot)
ggsave("observed_vs_predicted_plot.png", obs_pred_plot, width = 8, height = 6, dpi = 300)

# Heatmap 
numeric_data <- ml_data %>% select(where(is.numeric))
# Remove constant columns
numeric_data <- numeric_data[, sapply(numeric_data, function(x) sd(x) != 0)]

cor_matrix <- cor(numeric_data)
cor_df <- melt(cor_matrix)

cor_plot <- ggplot(cor_df, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       midpoint = 0, limit = c(-1,1), space = "Lab",
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        axis.title.x = element_blank(),
        axis.title.y = element_blank()) +
  coord_fixed() +
  labs(title = "Correlation Heatmap of Numeric Environmental Variables")

print(cor_plot)
ggsave("correlation_heatmap.png", cor_plot, width = 8, height = 6, dpi = 300)
