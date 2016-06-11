data {
  int D;                // Number of supernovae
  vector[D] ston;
  vector[D] fracincrease;
  int found[D];
}

transformed data {
  vector[D] lnFI;
  lnFI <- log(fracincrease);

  {
    real  minlnFI;
    real maxlnFI;

    minlnFI <- min(lnFI);
    maxlnFI <- max(lnFI);

    lnFI <- lnFI-minlnFI;
    lnFI <- lnFI/(maxlnFI-minlnFI);
  }
}
parameters {
  vector<lower=0.2, upper=1>[2] a;
  vector<lower=2, upper=5>[2] b;
  vector<lower=0.2, upper=5>[2] c;
}

model {
  # vector[D] a;
  # vector[D] b;
  # vector[D] c;

  vector[D] eff;

  # a<- a_[1] + (a_[2]-a_[1])* lnFI;
  # b<- b_[1] + (b_[2]-b_[1])* lnFI;
  # c<- c_[1] + (c_[2]-c_[1])* lnFI;

  # eff <- (ston-b) ./ c;
  eff <- (ston-(b[1] + (b[2]-b[1])* lnFI)) ./ (c[1] + (c[2]-c[1])* lnFI);



  # print (a,b,c);
  for (d in 1:D) { 
    if (eff[d] <=0){
      eff[d] <- machine_precision();
    } else{
      eff[d] <- tanh(eff[d]);
    }
    # eff[d] <- inv_logit(eff[d]);
  }
  eff <- (a[1] + (a[2]-a[1])* lnFI) .* eff;

  found ~ bernoulli(eff);
}