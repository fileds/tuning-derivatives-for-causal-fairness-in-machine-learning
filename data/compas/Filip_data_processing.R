library(tidyverse)
raw_data <- read.csv("~/dev/causality/fair-tuning/data/compas/compas-scores-two-years.csv")
nrow(raw_data)

# Filtering missing data as in Pro-Publica analysis
df <- raw_data |> select(age, c_charge_degree, race, age_cat, score_text, sex, priors_count, 
                    days_b_screening_arrest, decile_score, is_recid, two_year_recid, c_jail_in, c_jail_out) |> 
  filter(days_b_screening_arrest <= 30) |>
  filter(days_b_screening_arrest >= -30) |>
  filter(is_recid != -1) |>
  filter(c_charge_degree != "O") |>
  filter(score_text != 'N/A')
nrow(df)
df |> head()

# Save unaltered data
df |>
  filter(race %in% c("African-American", "Caucasian")) |>
  select(sex, age, race, priors_count, c_charge_degree, two_year_recid) |>
  write_csv("~/dev/causality/fair-tuning/data/compas/compas-scores-two-years-processed.csv")

# Mutating variable types as in Pro-Publica analysis 
df <- mutate(df, crime_factor = factor(c_charge_degree)) |>
  mutate(age_factor = as.factor(age_cat)) |>
  within(age_factor <- relevel(age_factor, ref = 1)) |>
  mutate(race_factor = factor(race)) |>
  within(race_factor <- relevel(race_factor, ref = 3)) |>
  mutate(gender_factor = factor(sex, labels= c("Female","Male"))) |>
  within(gender_factor <- relevel(gender_factor, ref = 2)) |>
  mutate(score_factor = factor(score_text != "Low", labels = c("LowScore","HighScore")))

df |> glimpse()

# Number of recidivism (same as in Pro-Publica analysis)
df |> count(two_year_recid)

# Modelling as in Pro-Publica analysis
model <- glm(score_factor ~ gender_factor + age_factor + race_factor +
               priors_count + crime_factor + two_year_recid, family="binomial", data=df)
summary(model)

# Exporting
# Age as factor
df |>
  filter(race_factor %in% c("African-American", "Caucasian")) |>
  select(score_factor, gender_factor, age_factor, race_factor, 
               priors_count, crime_factor, two_year_recid) |>
  write_csv("~/dev/causality/fair-tuning/data/compas/compas-scores-two-years-propub.csv")

# Filtering African-American and Caucasian defendants
df <- df |>
  filter(race_factor %in% c("African-American", "Caucasian")) |>
  within(
    race_factor <- relevel(race_factor, ref = 1)
  )
df |> glimpse()
model <- glm(score_factor ~ gender_factor + age_factor + race_factor +
               priors_count + crime_factor + two_year_recid, family="binomial", data=df)
summary(model)

model <- glm(is_recid ~ gender_factor + age_factor + race_factor +
               priors_count + crime_factor + two_year_recid, family="binomial", data=df)
summary(model)
