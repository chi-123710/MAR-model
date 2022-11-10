###estimate for the mar model
###use LSE&PROJ method
tenAR.est <- function(xx, R=1, P=1, method="LSE", init.A=NULL, init.sig=NULL, niter=150, tol=1e-6){
  if (identical("PROJ", method)) {
    tenAR.PROJ(xx, R, P)
  } else if (identical("LSE", method)) {
    tenAR.LS(xx, R, P, init.A, niter, tol, print.true=FALSE)
  }
}