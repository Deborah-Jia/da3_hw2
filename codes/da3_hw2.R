
# ------------------------------------------------------------------------------------------------------
#### SET UP

rm(list=ls())
options(scipen = 999)
# Import libraries
library(haven)
library(glmnet)
library(purrr)
library(margins)
library(skimr)
library(kableExtra)
library(Hmisc)
library(cowplot)
library(gmodels) 
library(lspline)
library(sandwich)
library(modelsummary)

library(rattle)
library(caret)
library(pROC)
library(ranger)
library(rpart)
library(partykit)
library(rpart.plot)


# set working directory
# option A: open material as project
# option B: set working directory for da_case_studies
#           example: setwd("C:/Users/bekes.gabor/Documents/github/da_case_studies/")

# set data dir, data used
source("set-data-directory.R")             # data_dir must be first defined 
# alternative: give full path here, 
#            example data_dir="C:/Users/bekes.gabor/Dropbox (MTA KRTK)/bekes_kezdi_textbook/da_data_repo"

# load theme and functions
source("ch00-tech-prep/theme_bg.R")
source("ch00-tech-prep/da_helper_functions.R")

data_in <- paste(data_dir,"da3_hw2/data", sep = "/")

#use_case_dir <- "ch17-predicting-firm-exit/"

data_out <- paste0(data_dir,"/da3_hw2/data/clean/", sep = "/")
output <- paste0(data_in,"clean/")


###########################################################
# Import data
###########################################################

data <- read.csv(paste(data_in,"raw/cs_bisnode_panel.csv", sep = "/"))

# drop variables with many NAs
data <- data %>%
  select(-c(COGS, finished_prod, net_dom_sales, net_exp_sales, wages)) %>%
  filter(year != 2016)


###########################################################
# label engineering
###########################################################
# add all missing year and comp_id combinations -
# originally missing combinations will have NAs in all other columns

data <- data %>%
  complete(year, comp_id)

# generate status_alive; if sales larger than zero and not-NA, then firm is alive
data  <- data %>%
  mutate(status_alive = sales > 0 & !is.na(sales) %>%
           as.numeric(.))


calc_cagr <- function(df, n) {
  df <- df %>%
    arrange(year, comp_id) %>%
    group_by(comp_id) %>%
    mutate(cagr = ((sales / lag(sales, n)) ^ (1 / n)) - 1)
  
  return(df)
}


data <- calc_cagr(data,2)

table(data$cagr)

# fast growth is when cagr is above 0.20, denoted with 
data <- data %>%
  mutate(fast_growth = ifelse( cagr >= 0.2, 1, 0) %>%
           as.numeric(.))
  


data <- data %>%
  filter(year <=2013)

# Size and growth
summary(data$sales)

data <- data %>%
  mutate(sales = ifelse(sales < 0, 1, sales),
         ln_sales = ifelse(sales > 0, log(sales), 0),
         sales_mil=sales/1000000,
         sales_mil_log = ifelse(sales > 0, log(sales_mil), 0))



table(is.nan(data$cagr))

# since dividing with 0 is not possible, 
# for these the cagr will be NaN, we replace them with 0
data$cagr <- ifelse(is.nan(data$cagr), 0, data$cagr )



data <- data %>%
  mutate(sales = ifelse(sales < 0, 1, sales),
         ln_sales = ifelse(sales > 0, log(sales), 0),
         sales_mil=sales/1000000,
         sales_mil_log = ifelse(sales > 0, log(sales_mil), 0))

data <- data %>%
  group_by(comp_id) %>%
  mutate(d1_sales_mil_log = sales_mil_log - Lag(sales_mil_log, 1) ) %>%
  ungroup()

data <- data %>%
  mutate(age = (year - founded_year) %>%
           ifelse(. < 0, 0, .),
         new = as.numeric(age <= 1) %>% #  (age could be 0,1 )
           ifelse(balsheet_notfullyear == 1, 1, .),
         d1_sales_mil_log = ifelse(new == 1, 0, d1_sales_mil_log),
         new = ifelse(is.na(d1_sales_mil_log), 1, new),
         d1_sales_mil_log = ifelse(is.na(d1_sales_mil_log), 0, d1_sales_mil_log))

# generate status_alive; if sales larger than zero and not-NA, then firm is alive
data  <- data %>%
  mutate(status_alive = sales > 0 & !is.na(sales) %>%
           as.numeric(.))

###########################################################
# sample design
###########################################################

# look at cross section
data <- data %>%
  filter((year == 2012) & (status_alive == 1)) %>%
  # look at firms below 10m euro revenues and above 1000 euros
  filter(!(sales_mil > 10)) %>%
  filter(!(sales_mil < 0.001))

Hmisc::describe(data$fast_growth)
summary(data$fast_growth)


data <- data %>% filter(!is.na(fast_growth))

write_csv(data,paste0(data_out,"work5.csv"))

###########################################################
# Feature engineering
###########################################################

data <- read_csv(paste0(data_out,"work5.csv"))

# change some industry category codes
data <- data %>%
  mutate(ind2_cat = ind2 %>%
           ifelse(. > 56, 60, .)  %>%
           ifelse(. < 26, 20, .) %>%
           ifelse(. < 55 & . > 35, 40, .) %>%
           ifelse(. == 31, 30, .) %>%
           ifelse(is.na(.), 99, .)
  )

table(data1$ind2_cat)

# Firm characteristics
data <- data %>%
  mutate(age2 = age^2,
         foreign_management = as.numeric(foreign >= 0.5),
         gender_m = factor(gender, levels = c("female", "male", "mix")),
         m_region_loc = factor(region_m, levels = c("Central", "East", "West")))

###########################################################
# look at more financial variables, create ratios
###########################################################

# assets can't be negative. Change them to 0 and add a flag.
data <-data  %>%
  mutate(flag_asset_problem=ifelse(intang_assets<0 | curr_assets<0 | fixed_assets<0,1,0  ))

table(data$flag_asset_problem)

data <- data %>%
  mutate(intang_assets = ifelse(intang_assets < 0, 0, intang_assets),
         curr_assets = ifelse(curr_assets < 0, 0, curr_assets),
         fixed_assets = ifelse(fixed_assets < 0, 0, fixed_assets))


# generate total assets
data <- data %>%
  mutate(total_assets_bs = intang_assets + curr_assets + fixed_assets)

summary(data$total_assets_bs)


pl_names <- c("extra_exp","extra_inc",  "extra_profit_loss", "inc_bef_tax" ,"inventories",
              "material_exp", "profit_loss_year", "personnel_exp")
bs_names <- c("intang_assets", "curr_liab", "fixed_assets", "liq_assets", "curr_assets",
              "share_eq", "subscribed_cap", "tang_assets" )

# divide all pl_names elements by sales and create new column for it
data <- data %>%
  mutate_at(vars(pl_names), funs("pl"=./sales))

# divide all bs_names elements by total_assets_bs and create new column for it
data <- data %>%
  mutate_at(vars(bs_names), funs("bs"=ifelse(total_assets_bs == 0, 0, ./total_assets_bs)))


########################################################################
# creating flags, and winsorizing tails
########################################################################

# Variables that represent accounting items that cannot be negative (e.g. materials)
zero <-  c("extra_exp_pl", "extra_inc_pl", "inventories_pl", "material_exp_pl", "personnel_exp_pl",
           "curr_liab_bs", "fixed_assets_bs", "liq_assets_bs", "curr_assets_bs", "subscribed_cap_bs",
           "intang_assets_bs")

data <- data %>%
  mutate_at(vars(zero), funs("flag_high"= as.numeric(.> 1))) %>%
  mutate_at(vars(zero), funs(ifelse(.> 1, 1, .))) %>%
  mutate_at(vars(zero), funs("flag_error"= as.numeric(.< 0))) %>%
  mutate_at(vars(zero), funs(ifelse(.< 0, 0, .)))


# for vars that could be any, but are mostly between -1 and 1
any <-  c("extra_profit_loss_pl", "inc_bef_tax_pl", "profit_loss_year_pl", "share_eq_bs")

data <- data %>%
  mutate_at(vars(any), funs("flag_low"= as.numeric(.< -1))) %>%
  mutate_at(vars(any), funs(ifelse(.< -1, -1, .))) %>%
  mutate_at(vars(any), funs("flag_high"= as.numeric(.> 1))) %>%
  mutate_at(vars(any), funs(ifelse(.> 1, 1, .))) %>%
  mutate_at(vars(any), funs("flag_zero"= as.numeric(.== 0))) %>%
  mutate_at(vars(any), funs("quad"= .^2))


# dropping flags with no variation
variances<- data %>%
  select(contains("flag")) %>%
  apply(2, var, na.rm = TRUE) == 0

data <- data %>%
  select(-one_of(names(variances)[variances]))

########################################################################
# additional
# including some imputation
########################################################################

# CEO age
data <- data %>%
  mutate(ceo_age = year-birth_year,
         flag_low_ceo_age = as.numeric(ceo_age < 25 & !is.na(ceo_age)),
         flag_high_ceo_age = as.numeric(ceo_age > 75 & !is.na(ceo_age)),
         flag_miss_ceo_age = as.numeric(is.na(ceo_age)))

data <- data %>%
  mutate(ceo_age = ifelse(ceo_age < 25, 25, ceo_age) %>%
           ifelse(. > 75, 75, .) %>%
           ifelse(is.na(.), mean(., na.rm = TRUE), .),
         ceo_young = as.numeric(ceo_age < 40))

# number emp, very noisy measure
data <- data %>%
  mutate(labor_avg_mod = ifelse(is.na(labor_avg), mean(labor_avg, na.rm = TRUE), labor_avg),
         flag_miss_labor_avg = as.numeric(is.na(labor_avg)))

summary(data$labor_avg)
summary(data$labor_avg_mod)

data <- data %>%
  select(-labor_avg)

# create factors
data <- data %>%
  mutate(urban_m = factor(urban_m, levels = c(1,2,3)),
         ind2_cat = factor(ind2_cat, levels = sort(unique(data$ind2_cat))))

data <- data %>%
  mutate(fast_growth_f = factor(fast_growth, levels = c(0,1)) %>%
           recode(., `0` = 'NO', `1` = "YES"))

########################################################################
# sales 
########################################################################

data <- data %>%
  mutate(sales_mil_log_sq=sales_mil_log^2)


ggplot(data = data, aes(x=sales_mil_log, y=as.numeric(fast_growth))) +
  geom_point(size=2,  shape=20, stroke=2, fill="blue", color="blue") +
  geom_smooth(method = "lm", formula = y ~ poly(x,2), color=color[4], se = F, size=1)+
  geom_smooth(method="loess", se=F, colour=color[5], size=1.5, span=0.9) +
  labs(x = "sales_mil_log",y = "fast growth") +
  theme_bg()


ols_s <- lm(fast_growth~sales_mil_log+sales_mil_log_sq,
            data = data)
summary(ols_s)

########################################################################
# sales change
########################################################################
# Note: graphs not in book

# lowess
Hmisc::describe(data$d1_sales_mil_log) # no missing

d1sale_1<-ggplot(data = data, aes(x=d1_sales_mil_log, y=as.numeric(fast_growth))) +
  geom_point(size=0.1,  shape=20, stroke=2, fill=color[2], color=color[2]) +
  geom_smooth(method="loess", se=F, colour=color[1], size=1.5, span=0.9) +
  labs(x = "Growth rate (Diff of ln sales)",y = "F ast growth") +
  theme_bg() +
  scale_x_continuous(limits = c(-6,10), breaks = seq(-10,10, 5))
d1sale_1
# save_fig("ch17-extra-1", output, "small")

# generate variables ---------------------------------------------------

data <- data %>%
  mutate(flag_low_d1_sales_mil_log = ifelse(d1_sales_mil_log < -1.5, 1, 0),
         flag_high_d1_sales_mil_log = ifelse(d1_sales_mil_log > 1.5, 1, 0),
         d1_sales_mil_log_mod = ifelse(d1_sales_mil_log < -1.5, -1.5,
                                       ifelse(d1_sales_mil_log > 1.5, 1.5, d1_sales_mil_log)),
         d1_sales_mil_log_mod_sq = d1_sales_mil_log_mod^2
  )

# no more imputation, drop obs if key vars missing
data <- data %>%
  filter(!is.na(liq_assets_bs),!is.na(foreign), !is.na(ind))

# drop missing
data <- data %>%
  filter(!is.na(age),!is.na(foreign), !is.na(material_exp_pl), !is.na(m_region_loc))

Hmisc::describe(data$age)

# drop unused factor levels
data <- data %>%
  mutate_at(vars(colnames(data)[sapply(data, is.factor)]), funs(fct_drop))

d1sale_2<-ggplot(data = data, aes(x=d1_sales_mil_log_mod, y=as.numeric(fast_growth))) +
  geom_point(size=0.1,  shape=20, stroke=2, fill=color[2], color=color[2]) +
  geom_smooth(method="loess", se=F, colour=color[1], size=1.5, span=0.9) +
  labs(x = "Growth rate (Diff of ln sales)",y = "fast_growth") +
  theme_bg() +
  scale_x_continuous(limits = c(-1.5,1.5), breaks = seq(-1.5,1.5, 0.5))
d1sale_2
# save_fig("ch17-extra-2", output, "small")

d1sale_3<-ggplot(data = data, aes(x=d1_sales_mil_log, y=d1_sales_mil_log_mod)) +
  geom_point(size=0.1,  shape=20, stroke=2, fill=color[2], color=color[2]) +
  labs(x = "Growth rate (Diff of ln sales) (original)",y = "Growth rate (Diff of ln sales) (winsorized)") +
  theme_bg() +
  scale_x_continuous(limits = c(-5,5), breaks = seq(-5,5, 1)) +
  scale_y_continuous(limits = c(-3,3), breaks = seq(-3,3, 1))
d1sale_3
# save_fig("ch17-extra-3", output, "small")



# check variables
# datasummary_skim(data, type="numeric")

write_csv(data,paste0(data_out,"bisnode_firms_clean.csv"))
write_rds(data,paste0(data_out,"bisnode_firms_clean.rds"))

