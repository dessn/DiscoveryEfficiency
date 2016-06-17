data {
  int D;                // Number of supernovae
  vector[D] ston;
  vector<lower=0>[D] fracincrease;
  int found[D];
}

transformed data {
  vector[D] lnFI;
  vector[D] oneminuslnFIto4th;
  lnFI <- log(fracincrease);


  {
    real  minlnFI;
    real maxlnFI;

    minlnFI <- min(lnFI);
    maxlnFI <- max(lnFI);

    lnFI <- lnFI-minlnFI;
    lnFI <- lnFI/(maxlnFI-minlnFI);
  }

  for (d in 1:D) {   
    oneminuslnFIto4th[d] <- (1- lnFI[d])^4;
  }

}
parameters {
  # vector<lower=0.2, upper=6>[2] a;
  real<lower=0.2, upper=1> a1;
  real<lower=1, upper=6> a2;
  vector<lower=1, upper=6>[2] b;
  vector<lower=0.5, upper=8>[2] c;
  real <lower=.5, upper =5> pop;
}

model {

  vector[D] eff;

  eff <- (ston-(b[1] + (b[2]-b[1])* lnFI)) ./ (c[1] + (c[2]-c[1])* oneminuslnFIto4th);

  for (d in 1:D) { 
    if (eff[d] <=0){
      eff[d] <- machine_precision();
    } else {
      eff[d] <- exp(-a1*((1-lnFI[d])^a2))* tanh(eff[d])^pop;
    }
  }

  found ~ bernoulli(eff);
}