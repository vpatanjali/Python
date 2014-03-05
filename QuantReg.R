setwd('G:/Kaggle/Default/')

require(quantreg)

data = read.csv('train_v2_log_rlu_scale_impute_dev.csv', header = FALSE)
dvs = read.csv('train_v2_log_rlu_scale_impute_dev_dvs.csv', header = FALSE)
#data = read.csv('test.csv', header = FALSE)
#dvs = read.csv('test2.csv', header = FALSE)

names(dvs) = c('dv')

selectedcols = array(TRUE,dim(data)[2])

for(i in 1:dim(data)[2]){selectedcols[i] = length(unique(data[,i]))>1}
print(sum(selectedcols))
data = data[,which(selectedcols)]
data = data[,which(selectedcols)]
removecols = duplicated(t(data))
print(sum(removecols))
data = data[,which(!removecols)]

top_10 = irlba(data,nu=10,nv=10)

data_10 = data%*%top_10$v
colnames(data_10) = paste('pc',1:10, sep='')

dataframe = cbind(data_10, dvs)

formula = as.formula(paste('dv', paste(colnames(dataframe)[-dim(dataframe)[2]], sep = '', collapse = ' + '), sep = '~'))

#dataframe1 = dataframe[1:1000,]

model = rq(formula, 0.5, dataframe)