##predict，use either LSE or MLE method.You can set the rolling window size.
##example:# prediction for reduced rank MAR(1) model
##est <- tenAR.estt(xx, method="LSE")
##pred <- tenAR.predict(est, n.ahead = 1)
#rolling forcast
##set the rolling window size
##n0 = t - min(50,t/2)
##pred.rolling <- tenAR.predict(est, n.ahead = 5, rolling=TRUE, n0=n0)

tenAR.predict <- function(object, n.ahead=1, xx=NULL, rolling=FALSE, n0=NULL){
  if (is.null(object$SIGMA)){method = "LSE"} else {method = "MLE"}
  if (!is.null(object$A1)){
    A <- list(list(list(object$A1, object$A2)))
    if (is.null(object$Sig1)){
      method = "RRLSE"
    } else {
      method = "RRMLE"
    }
  } else if (!is.null(object$coef)){
    A <- list(list(object$coef))
    method = "VAR"
  } else {
    A <- object$A
  }
  if (is.null(xx)){xx = object$data}
  print(mode(xx))
  if (mode(xx) != "S4") {xx <- rTensor::as.tensor(xx)}
  if (rolling == TRUE){
    return(predict.rolling(A, xx, n.ahead, method, n0))
  }
  P <- length(A)
  R <- sapply(c(1:P), function(l){length(A[[l]])})
  K <- xx@num_modes - 1
  dim <- xx@modes
  ttt <- (dim[1]+1):(dim[1]+n.ahead)
  for(tt in ttt){
    L1 = 0
    for (l in c(1:P)){
      if (R[l] == 0) next
      L1 <- L1 + Reduce("+",lapply(c(1:R[l]), function(n) {rTensor::ttl(abind::asub(xx, tt-l, 1, drop=FALSE), A[[l]][[n]], (c(1:K) + 1))}))
    }
    xx <- as.tensor(abind(xx@data, L1@data, along=1))
  }
  return(abind::asub(xx@data, ttt, 1, drop=FALSE))
}


predict.rolling <- function(A, xx, n.ahead, method, n0){
  if ((method == "RRLSE") || (method == "RRMLE")){
    k1 <- rankMatrix(A[[1]][[1]][[1]])
    k2 <- rankMatrix(A[[1]][[1]][[2]])
  }
  P <- length(A)
  R <- sapply(c(1:P), function(l){length(A[[l]])})
  K <- xx@num_modes - 1
  dim <- xx@modes
  t <- dim[1]
  if(is.null(n0)){n0 = t - min(50,t/2)}
  ttt <- (n0):(t - n.ahead)
  for(tt in ttt){
    tti <- tt - ttt[1] + 1
    print(paste("rolling forcast t =", tti))
    if (method == "RRLSE"){
      model <- MAR1.RR(abind::asub(xx@data, 1:tt, 1, drop=FALSE), k1, k2)
      A <- list(list(list(model$A1, model$A2)))
    } else if (method == "RRMLE"){
      model <- MAR1.CC(abind::asub(xx@data, 1:tt, 1, drop=FALSE), k1, k2)
      A <- list(list(list(model$A1, model$A2)))
    } else {
      model = tenAR.est(abind::asub(xx@data, 1:tt, 1, drop=FALSE), R, c(P,0), method)
      A <- model$A
    }
    L1 = 0
    for (l in c(1:P)){
      if (R[l] == 0) next
      L1 <- L1 + Reduce("+",lapply(c(1:R[l]), function(n) {rTensor::ttl(abind::asub(xx, tt-l+1, 1, drop=FALSE), A[[l]][[n]], (c(1:K) + 1))}))
    }
    if (tti == 1){xx.pred = L1@data} else {xx.pred = abind(xx.pred, L1@data, along=1)}
  }
  return(xx.pred)
}

