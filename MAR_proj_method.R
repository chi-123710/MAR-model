###This is the projection method for initialization of mar model.
###You can also use library(tensorTS) and type the tenAR.est(a,method="PROJ")
##lag 1 projection
MAR1.PROJ <- function(xx){
  # xx: T * p * q
  # X_t = LL X_{t-1} RR + E_t
  # Sig = cov(vec(E_t))
  # one-step projection estimation
  # Return LL, RR, and estimate of Sig
  dd <- dim(xx)
  T <- dd[1]
  p <- dd[2]
  q <- dd[3]
  xx.mat <- matrix(xx,T,p*q)
  ###projection to the kroneck dimension
  kroneck <- t(xx.mat[2:T,]) %*% xx.mat[1:(T-1),] %*% solve(t(xx.mat[1:(T-1),]) %*% xx.mat[1:(T-1),])
  ans.projection <- projection(kroneck,r=1,p,q,p,q)
  ###use svd to solve for the equation
  a <- svd(ans.projection[[1]][[1]],nu=0,nv=0)$d[1]
  ###the solution for the left matrix,since after derivation it is casual
  ### to divide a constant and the solution not change
  LL <- ans.projection[[1]][[1]] / a
  ###the solution for the right matrix
  RR <- t(ans.projection[[1]][[2]]) * a
  ###calculate for the residual
  res = xx[2:T,,,drop=FALSE] - aperm(tensor(tensor(xx[1:(T-1),,,drop=FALSE],RR,3,1),LL,2,2),c(1,3,2))
  Sig <- matrix(tensor(res,res,1,1),p*q)/(T-1)
  # return the solution A,B and error matrix
  
  return(list(A1=LL,A2=RR,Sig=Sig))
}
###lag 2 projection
MAR2.PROJ <- function(xx, R=1, P=1){
  # xx: T * p * q
  # X_t = LL X_{t-1} RR + E_t
  # Sig = cov(vec(E_t))
  # one-step projection estimation
  # Return LL, RR, and estimate of Sig
  dd <- dim(xx)
  T <- dd[1]
  d1 <- dd[2]
  d2 <- dd[3]
  # xx.mat <- matrix(xx,T,d1*d2)
  # kroneck <- t(xx.mat[2:T,]) %*% xx.mat[1:(T-1),] %*% solve(t(xx.mat[1:(T-1),]) %*% xx.mat[1:(T-1),])
  A = list()
  kroneck <- tenAR.VAR(xx, P)$coef
  for (i in c(1:P)){
    ans.projection <- projection(kroneck[[i]],R[i],d2,d1,d2,d1)
    for (j in c(1:R[i])){
      a = svd(ans.projection[[j]][[1]],nu=0,nv=0)$d[1]
      ans.projection[[j]][[1]] <- ans.projection[[j]][[1]] / a
      ans.projection[[j]][[2]] <- t(ans.projection[[j]][[2]]) * a
    }
    A[[i]] <- ans.projection
  }
  return(list(A=A))
}