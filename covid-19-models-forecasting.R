library(tidyverse)
library(tidymodels)
library(modeltime)
library(timetk)   
library(lubridate)
library(readr)
library(dplyr)
library(tidyr)
library(glmnet)
library(randomForest)

data_daily  <- read.csv("~/Workdir/COVID-19-master/data/owid-covid-data.csv")
data_daily <- data_daily[data_daily$location == 'France',]

data_daily <- data_daily %>% select(date, new_cases) %>% drop_na(new_cases)

time_series_covid19_confirmed_global <- aggregate(data_daily$new_cases, by=list(date=data_daily$date), FUN=sum)

covid19_confirmed_global_tbl <- time_series_covid19_confirmed_global %>%
  select(date, x) %>%
  rename(value=x) 

covid19_confirmed_global_tbl$date <- as.Date(covid19_confirmed_global_tbl$date, format = "%Y-%m-%d")

# covid19_confirmed_global_tbl <- covid19_confirmed_global_tbl[covid19_confirmed_global_tbl[["date"]] >= "2020-01-17", ]
covid19_confirmed_global_tbl <- covid19_confirmed_global_tbl[covid19_confirmed_global_tbl[["date"]] >= "2020-02-26", ]

covid19_confirmed_global_tbl <- as.tibble(covid19_confirmed_global_tbl)

covid19_confirmed_global_tbl %>%
  plot_time_series(date, value, .interactive = FALSE)


splits <- covid19_confirmed_global_tbl %>%
  time_series_split(assess = "5 days", cumulative = TRUE)

splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, value, .interactive = FALSE)

model_fit_arima <- arima_reg() %>%
  set_engine("auto_arima") %>%
  fit(value ~ date, training(splits))

model_fit_arima

model_fit_prophet <- prophet_reg() %>%
  set_engine("prophet", yearly.seasonality = TRUE) %>%
  fit(value ~ date, training(splits))

model_fit_prophet


recipe_spec <- recipe(value ~ date, training(splits)) %>%
  step_timeseries_signature(date) %>%
  step_rm(contains("am.pm"), contains("hour"), contains("minute"),
          contains("second"), contains("xts")) %>%
  step_fourier(date, period = 365, K = 5) %>%
  step_dummy(all_nominal())

recipe_spec %>% prep() %>% juice()

model_spec_glmnet <- linear_reg(penalty = 0.01, mixture = 0.5) %>%
  set_engine("glmnet")

workflow_fit_glmnet <- workflow() %>%
  add_model(model_spec_glmnet) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  fit(training(splits))

model_spec_rf <- rand_forest(trees = 500, min_n = 50) %>%
  set_engine("randomForest")

workflow_fit_rf <- workflow() %>%
  add_model(model_spec_rf) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  fit(training(splits))

model_spec_prophet_boost <- prophet_boost() %>%
  set_engine("prophet_xgboost", yearly.seasonality = TRUE) 

workflow_fit_prophet_boost <- workflow() %>%
  add_model(model_spec_prophet_boost) %>%
  add_recipe(recipe_spec) %>%
  fit(training(splits))

workflow_fit_prophet_boost


model_table <- modeltime_table(
  model_fit_arima, 
  model_fit_prophet,
  workflow_fit_glmnet,
  workflow_fit_rf,
  workflow_fit_prophet_boost
) 

model_table


calibration_table <- model_table %>%
  modeltime_calibrate(testing(splits))

calibration_table


calibration_table %>%
  modeltime_forecast(actual_data = covid19_confirmed_global_tbl) %>%
  plot_modeltime_forecast(.interactive = FALSE)

calibration_table %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)


calibration_table %>%
  # Remove ARIMA model with low accuracy
  # filter(.model_id != 1) %>%
  
  # Refit and Forecast Forward
  modeltime_refit(covid19_confirmed_global_tbl) %>%
  modeltime_forecast(h = "3 months", actual_data = covid19_confirmed_global_tbl) %>%
  plot_modeltime_forecast(.interactive = FALSE)
