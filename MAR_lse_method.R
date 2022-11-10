###LSE you can also use tenAR.est(a,method="LSE")
MAR1.LS <- function(xx,niter=50,tol=1e-6,print.true = FALSE){
  # xx: T * p * q
  # X_t = LL X_{t-1} RR + E_t
  # Sig = cov(vec(E_t))
  # LS criterion
  # iterative algorithm between LL <--> RR
  # Return LL, RR, and estimate of Sig
  dd=dim(xx)
  T <- dd[1]
  p <- dd[2]
  q <- dd[3]
  LL.old <- diag(p)
  RR.old <- diag(q)
  dis <- 1
  iiter <- 1

  while(iiter <= niter & dis >= tol){
    # estimate RR0 use kronecker product
    temp <- tensor(xx[1:(T-1),,,drop=FALSE],LL.old,2,2)  # (T-1) * q * p
    AA <- tensor(temp,temp,c(1,3),c(1,3))
    BB <- tensor(temp,xx[2:T,,,drop=FALSE],c(1,3),c(1,2))
    RR <- solve(AA,BB)
    # estimate LL0 use kronecker product
    temp <- tensor(xx[1:(T-1),,,drop=FALSE],RR,3,1)  # (T-1) * p * q
    AA <- tensor(temp,temp,c(1,3),c(1,3))
    BB <- t(tensor(temp,xx[2:T,,,drop=FALSE],c(1,3),c(1,3)))
    LL <- t(solve(t(AA),t(BB)))
    a <- svd(LL,nu=0,nv=0)$d[1]
    LL <- LL / a
    RR <- RR * a
    # update for the next iteration
    dis <- sqrt(sum((kronecker(t(RR),LL)-kronecker(t(RR.old),LL.old))^2))
    LL.old <- LL
    RR.old <- RR
    iiter <- iiter + 1
    if(print.true==TRUE){
      print(LL)
      print(RR)
    }
  }
  ###calculate for the residual
  res=xx[2:T,,,drop=FALSE] - aperm(tensor(tensor(xx[1:(T-1),,,drop=FALSE],RR,3,1),LL,2,2),c(1,3,2))
  Sig <- matrix(tensor(res,res,1,1),p*q)/(T-1)
  sd <- MAR.SE(xx, t(RR), LL, Sig)
  return(list(A1=LL,A2=t(RR),res=res,Sig=Sig,niter=iiter,sd=sd))
}
