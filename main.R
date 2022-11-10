###The most important file!!!
##You can use it to transfer the csv file produced from Dateaset.py
install.packages("tensorTS")
library(stringr)
library(tensorTS)
library(magrittr)
#read the csv file
df0 = read.csv('./estate.csv', header = F)
d = c()
##transfer based on the col and row
for(y in 1:ncol(df0)){
  # dy = data.frame()
  for(x in 1:nrow(df0)){
    a = str_split(df0[x, y], '\n')[[1]] %>% str_c(collapse = '') %>% str_remove_all('\\[|\\]')
    b = str_split(a, ' ')[[1]]
    c = b[!(b %in% c('[', ']', ''))]
    # dy = rbind(dy, c)
    d = c(d, c)
  }
}
###rearrange the array 
m = array(d, dim=c(70,63,40))
##check for the dimension,notice now the third dimension is time t
m[, , 1] %>% dim()
##set the first dimension to be time t
tran_m=aperm(m,c(3,1,2))
##check
dim(tran_m)
##since we only test for 3*3 (in specific we check for 3 companies with
##three indicators,feel free to change it )
cut_m=tran_m[,1:3,1:3]
##numeric it 
c=as.numeric(cut_m)
##rearrange it,still,the first dimension is time
a=array(c,dim=c(40,3,3))
est=tenAR.est(a,method="LSE")
library(reticulate)
np=import("numpy")
est$A[[1]][[1]][[1]]
est$A[[1]][[1]][[2]]
a=np$array(est$A[[1]][[1]][[1]])
np$save("A_1.npy" , a)
b=np$array(est$A[[1]][[1]][[2]])
np$save("B_1.npy" , b)