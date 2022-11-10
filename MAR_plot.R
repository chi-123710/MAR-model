##plot for the time series and acf 
##use the transfer_r function to transfer the csv format to R base.
mplot <- function(xx){
  if (mode(xx) == "S4"){xx = xx@data}
  dim = dim(xx)
  time = array(c(1:dim[1]),dim[1])
  opar <- par(mfrow=c(dim[2],dim[3]),mai=0.05*c(1,1,1,1),oma=c(2,2,0,0))
  on.exit(par(opar))
  for(i in 1:dim[2]){
    for(j in 1:dim[3]){
      if(i!=dim[2] & j!=1){
        plot(time,xx[,i,j],type='l',xaxt='n',yaxt='n',ylim=range(xx[,i,]))
      }
      if(i!=dim[2] & j==1){
        plot(time,xx[,i,j],type='l',xaxt='n',ylim=range(xx[,i,]))
      }
      if(i==dim[2] & j!=1){
        plot(time,xx[,i,j],type='l',yaxt='n',ylim=range(xx[,i,]))
      }
      if(i==dim[2] & j==1){
        plot(time,xx[,i,j],type='l',ylim=range(xx[,i,]))
      }
    }
  }

}

##ACF
mplot.acf <- function(xx){
  if (mode(xx) == "S4"){xx = xx@data}
  dim = dim(xx)
  opar <- par(mfrow=c(dim[2],dim[3]),mai=0.05*c(1,1,1,1),oma=c(2,2,0,0))
  on.exit(par(opar))
  for(i in 1:dim[2]){
    for(j in 1:dim[3]){
      if(i!=dim[2] & j!=1){
        acf(xx[,i,j])
      }
      if(i!=dim[2] & j==1){
        acf(xx[,i,j])
      }
      if(i==dim[2] & j!=1){
        acf(xx[,i,j])
      }
      if(i==dim[2] & j==1){
        acf(xx[,i,j])
      }
    }
  }
}
