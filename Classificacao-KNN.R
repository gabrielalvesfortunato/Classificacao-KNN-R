# Classificação KNN em R


# Definindo um problema para classificação Binária
# American stock market index (NYSE ou NASDAQ)


# Definindo o diretorio de trabalho
setwd("C:\\Users\\Gabriel\\Desktop\\Cursos\\MachineLearning\\Cap06-KNN")
getwd()

# Instalando os pacotes
install.packages("ISLR")
install.packages("caret")
install.packages("e1071")

# Carregando os dados
library(ISLR)
library(caret)
library(e1071)

# Definindo o seed
set.seed(300)

# Carregando o conjunto de dados
?Smarket
summary(Smarket)
str(Smarket)
View(Smarket)

# Split do dataset em treino e teste
?createDataPartition
indxTrain <- createDataPartition(y = Smarket$Direction, p = 0.75, list = FALSE)
dados_treino <- Smarket[indxTrain,]
dados_teste <- Smarket[-indxTrain,]
class(dados_treino)
class(dados_teste)

# Verificando a ditribuição dos dados originais e das partições
prop.table(table(Smarket$Direction)) * 100
prop.table(table(dados_treino$Direction)) * 100

# Correlação entre as variáveis preditoras
descrCor <- cor(dados_treino[, names(dados_treino) != "Direction"])
descrCor 

# Normalização
scale.features <- function(df, variables) {
  for(variable in variables){
    df[[variable]] <- scale(df[[variable]], center = T, scale = T)
  }
  return(df)
}

# Removendo a variável target dos dados de treino e teste
numeric.vars_treino <- colnames(treinoX <- dados_treino[,names(dados_treino) != "Direction"])
numeric.vars_teste <- colnames(testeX <- dados_teste[,names(dados_teste) != "Direction"])

# Aplicando normalização às variaveis preditoras de treino e de teste
dados_treino_scaled <- scale.features(dados_treino, numeric.vars_treino)
dados_teste_scaled <- scale.features(dados_teste, numeric.vars_teste)

# Visualizando os novos dados normalizados
View(dados_teste_scaled)
View(dados_treino_scaled)

# Arquivo de controle (corss-validation)
ctrl <- trainControl(method = "repeatedcv", repeats = 3)

# Criação do modelo
knn_v1 <- train( Direction ~ .,
                 data = dados_treino_scaled,
                 method = "knn",
                 trControl = ctrl,
                 tuneLength = 20 )

# Avaliação do Modelo
knn_v1

# Numero de vizinhos X acuracia
plot(knn_v1)

# Fazendo Previsoes
knnPredict <- predict(knn_v1, newdata = dados_teste_scaled)
knnPredict

# Criando a confusion matrix
confusionMatrix(knnPredict, dados_teste_scaled$Direction)


### Aplicando outras Metricas ###


# Arquivo de controle
ctrl <- trainControl( method = "repeatedcv",
                      repeats = 3,
                      classProbs = TRUE,
                      summaryFunction = twoClassSummary )

# Treinamento do modelo
knn_v2 <- train( Direction ~ .,
                 data = dados_treino_scaled,
                 method = "knn", 
                 trControl = ctrl,
                 metric = "ROC",
                 tuneLength = 20 )

# Modelo KNN
knn_v2

# Numero de vizinhos mais proximos vs acuracia
plot(knn_v2)

# Fazendo as previsoes
knnPredict <- predict(knn_v2, newdata = dados_teste_scaled)

# Criando a Confusion Matrix
confusionMatrix(knnPredict, dados_teste_scaled$Direction)


## Previsões com novos dados ##


# Preparando dados de entrada
Year = c(2006, 2007, 2008)
Lag1 = c(1.30, 0.09, -0.654)
Lag2 = c(1.483, -0.198, 0.589)
Lag3 = c(-0.345, 0.029, 0.690)
Lag4 = c(1.398, 0.104, 1.483)
Lag5 = c(0.214, 0.105, 0.589)
Volume = c(1.36890, 1.09876, 1.231233)
Today = c(0.289, -0.497, 1.649)

novos_dados = data.frame(Year, Lag1, Lag2, Lag3, Lag4, Lag5, Volume, Today)
novos_dados
str(novos_dados)
class(novos_dados)

# normalizando os dados
novos_dados_scaled <- scale.features(novos_dados, colnames(novos_dados))
str(novos_dados_scaled)
class(novos_dados_scaled)

# Fazendo novas previsoes
knnPredict <- predict(knn_v2, novos_dados_scaled)
cat(sprintf("\n Previsão de \"%s\" é \"%s\"\n", novos_dados$Year, knnPredict))
