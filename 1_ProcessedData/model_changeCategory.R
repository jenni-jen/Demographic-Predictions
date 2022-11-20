setwd('')

train = read.csv("train2.csv",stringsAsFactors = F, fileEncoding='utf-8')
validation = read.csv("validation2.csv",stringsAsFactors = F, fileEncoding='utf-8')
train$brand_factor = as.factor(train$brand_factor)
validation$brand_factor = as.factor(validation$brand_factor)


# ===== logistic regression =====

# 1st fit
logistic_fit1 = glm(gender~events_holi_day+events_holi_night+events_work_day+events_work_night+
                     education+entertainment+finance+game+health+life+news+shopping+social+
                     tools+travel+brand_factor, data=train, family='binomial')
summary(logistic_fit1)

valid_4_logistic = validation[! validation$brand_factor %in% c(18, 31),]

get_accuracy_logi = function(fit, valid){
  pred = predict(fit, valid)
  pred_prop = exp(pred)/(1+exp(pred))
  pred_result = pred_prop>0.5
  pred_result = as.numeric(pred_result)
  
  accuracy = sum(pred_result == valid$gender)/dim(valid)[1]
  return(accuracy)
}

get_accuracy_logi(logistic_fit1, valid_4_logistic) # accuracy = 0.7097863


# 2nd fit: do not consider brand_factor
logistic_fit2 = glm(gender~events_holi_day+events_holi_night+events_work_day+
                      education+entertainment+finance+game+health+life+news+shopping+social+
                      tools, data=train, family='binomial')
summary(logistic_fit2)

get_accuracy_logi(logistic_fit2, valid_4_logistic) # accuracy = 0.7142857



# ===== random forest =====

library(randomForest)
set.seed(123)

my_forest1 = randomForest(as.factor(gender)~events_holi_day+events_holi_night+events_work_day+
                            events_work_night+education+entertainment+finance+game+health+
                            life+news+shopping+social+tools+travel,
                          data=train,importance=T)

importance(my_forest1,type=1)
# MeanDecreaseAccuracy: 拿掉那个值之后预测准确性下降多少（也可能有负的）

validation_dt=validation[,c('events_holi_day', 'events_holi_night', 'events_work_day',
                            'events_work_night', 'education', 'entertainment', 
                            'finance', 'game', 'health', 'life', 'news', 'shopping', 
                            'social', 'tools', 'travel')]
p_forest1=predict(my_forest1,validation_dt)

sum(p_forest1==validation$gender)/dim(validation)[1] # accuracy = 0.7138047


# for loop
ntree_v=c(200,300,400,500)
mtry_v=c(2,3,4,5)
tree_accuracy=c()

for(i in ntree_v){
  for(j in mtry_v){
    set.seed(123)
    my_forest_loop = randomForest(as.factor(gender)~events_holi_day+events_holi_night+events_work_day+
                                    events_work_night+education+entertainment+finance+game+health+
                                    life+news+shopping+social+tools+travel,
                                  data=train,ntree=i,mtry=j,importance=T)
    p_forest=predict(my_forest_loop,validation_dt)
    x=sum(p_forest==validation$gender)/length(validation$gender)
    
    tree_accuracy=c(tree_accuracy,x)
  }
}

which.max(tree_accuracy) # 第5次accuracy最高，即ntree=300, mtry=2
tree_accuracy[5] # accuracy = 0.7171717

set.seed(123)
my_forest_best = randomForest(as.factor(gender)~events_holi_day+events_holi_night+events_work_day+
                                events_work_night+education+entertainment+finance+game+health+
                                life+news+shopping+social+tools+travel,
                              data=train,ntree=300,mtry=2,importance=T)
p_forest=predict(my_forest_best,validation_dt)

sum(p_forest==validation$gender)/length(validation$gender) # accuracy = 0.7171717



# ===== XGBoost =====

library(xgboost)

trainXG=train[,c('events_holi_day', 'events_holi_night', 'events_work_day',
                 'events_work_night', 'education', 'entertainment', 
                 'finance', 'game', 'health', 'life', 'news', 'shopping', 
                 'social', 'tools', 'travel', 'brand_factor')]

set.seed(123)
gender_xg = xgboost(data = data.matrix(trainXG), label = train$gender, 
                    max.depth = 3, eta = 0.35,  nrounds = 45, objective = "binary:logistic")

valid_select = valid_4_logistic
validationXG = valid_select[,c('events_holi_day', 'events_holi_night', 'events_work_day',
                               'events_work_night', 'education', 'entertainment', 
                               'finance', 'game', 'health', 'life', 'news', 'shopping', 
                               'social', 'tools', 'travel', 'brand_factor')]
  
pred_xgb = predict(gender_xg, data.matrix(validationXG))
pred_xgb=pred_xgb>0.5
sum(pred_xgb==validation$gender)/length(validation$gender) # acurracy = 0.6857464



# ===== Neural Network =====

library('neuralnet')

# hidden layer = 3
set.seed(123)
nn=neuralnet(gender~events_holi_day+events_holi_night+events_work_day+
               events_work_night+education+entertainment+finance+game+health+
               life+news+shopping+social+tools+travel,
             data=train, hidden=3, act.fct = "logistic", linear.output = FALSE, stepmax=1e6)

plot(nn)
Predict=compute(nn,validation_dt) # nn的predict用compute函数
Predict$net.result

nn_pred=(Predict$net.result>0.5)+0 # 把bool变成0和1

sum(nn_pred==validation$gender)/length(validation$gender) # accuracy = 0.7025814


# hidden layer = 5
set.seed(123)
nn5=neuralnet(gender~events_holi_day+events_holi_night+events_work_day+
                events_work_night+education+entertainment+finance+game+health+
                life+news+shopping+social+tools+travel,
              data=train, hidden=5, act.fct = "logistic", linear.output = FALSE, stepmax=1e6)

plot(nn5)
Predict=compute(nn5,validation_dt) # nn的predict用compute函数
Predict$net.result

nn_pred=(Predict$net.result>0.5)+0 # 把bool变成0和1

sum(nn_pred==validation$gender)/length(validation$gender) # accuracy = 0.7126824


# hidden layer = 2,3
set.seed(123)
nn23=neuralnet(gender~events_holi_day+events_holi_night+events_work_day+
                 events_work_night+education+entertainment+finance+game+health+
                 life+news+shopping+social+tools+travel,
               data=train, hidden=c(2,3), act.fct = "logistic", linear.output = FALSE, stepmax=1e6)

plot(nn23)
Predict=compute(nn23,validation_dt) # nn的predict用compute函数
Predict$net.result

nn_pred=(Predict$net.result>0.5)+0 # 把bool变成0和1

sum(nn_pred==validation$gender)/length(validation$gender) # accuracy = 0.7138047







