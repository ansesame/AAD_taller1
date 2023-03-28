# Este documento contiene el código requerido para el desarrollo del taller 1
# del curso de Análisis Avanzado de datos de la maestría MACC en la universidad
# del Rosario.
# Código desarrollado por ...
# Marzo 2023
################################################################################
rm(list = ls())
SEED <- 1
# Dependencies
# install.packages("glmnet")

library(glmnet)
# Load data
gen_data <- read.csv('data/taller1.txt')
data_x <- gen_data[,2:ncol(gen_data)]

print('Descriptivas variables respuesta')
summary(gen_data$y)

cat('Número de observaciones:', ncol(data_x))
cat('Número de variables:', nrow(data_x))


# Dividir en entrenamiento y prueba
set.seed(SEED)
num_obs <- nrow(gen_data)
train_idx <- sample(num_obs, 1000)
test_idx <- seq(num_obs)[!seq(num_obs) %in% train_idx]

train_data <- gen_data[train_idx,]
test_data <- gen_data[test_idx,]

# Elegir mejores parámetros de Ridge y Lasso por validación cruzada.
num_folds = 10

# Ridge
l_seq_r <- seq(0.00001, 2, length = 100)
cv_ridge <- cv.glmnet(as.matrix(train_data[,-1]), train_data$y, 
                      alpha = 0, lambda = l_seq_r,
                      nfolds = num_folds, type.measure = "mse")
plot(cv_ridge$lambda, cv_ridge$cvm,
     xlab = "Lambda", ylab = "ECM", main = "CV - Ridge")
lambda_r <- cv_ridge$lambda.min

# Lasso
l_seq_l <- seq(0.0001, 1, length = 100)
cv_lasso <- cv.glmnet(as.matrix(train_data[,-1]), train_data$y, 
                      alpha = 1, lambda = l_seq_l,
                      nfolds = num_folds, type.measure = "mse")
plot(cv_lasso$lambda, cv_lasso$cvm,
     xlab = "Lambda", ylab = "ECM", main = "CV - Lasso")
lambda_l <- cv_lasso$lambda.min

# Entrenar con los mejores lambda
mod_ridge <- glmnet(as.matrix(train_data[,-1]), train_data$y, 
                    alpha = 0, lambda = lambda_r)
mod_lasso <- glmnet(as.matrix(train_data[,-1]), train_data$y, 
                    alpha = 1, lambda = lambda_l)

# Elegir entre ridge y lasso
pred_ridge <- predict(mod_ridge, as.matrix(test_data[,-1]))
pred_lasso <- predict(mod_lasso, as.matrix(test_data[,-1]))

cat("ECM - Ridge:", mean(pred_ridge - test_data$y)^2)           
cat("ECM - Lasso:", mean(pred_lasso - test_data$y)^2)

# Reentrenar con todos los datos
mod_l_final <- glmnet(as.matrix(data_x), gen_data$y, 
                      alpha = 1, lambda = lambda_l)

coefs_l <- as.data.frame(as.array(coef(mod_l_final)))
coefs_nz <- subset(coefs_l, s0 != 0)
cat("Número de variables relevantes:", nrow(coefs_nz)-1)

fig <- barplot(coefs_nz$s0, main='Coeficientes diferentes de 0',
               ylim=c(0, 1), ylab = 'Magnitud coeficiente')
axis(1, at = fig, labels = rownames(coefs_nz), lwd = 0)

# Trasa de los coeficientes
mod_l_test <- glmnet(as.matrix(data_x), gen_data$y, 
                      alpha = 1, lambda = l_seq_l)
plot(mod_l_test, xvar="lambda", label=TRUE,
     main = "",
     xlab = "log(Lambda)", ylab = "Coeficientes")
axis(side = 3,
     at = seq(par("usr")[1], par("usr")[2], len = 1000), 
     tck = -0.5,
     lwd = 2, 
     col = "white", 
     labels = F)