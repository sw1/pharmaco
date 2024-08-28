library(tidyverse)
library(stm)
library(tidytext)
library(LDAvis)
library(tm)

createJSON_noreorder <- function (phi = matrix(), theta = matrix(), doc.length = integer(), 
                                  vocab = character(), term.frequency = integer(), R = 30, 
                                  lambda.step = 0.01, mds.method = jsPCA, cluster, plot.opts = list(xlab = "PC1", 
                                                                                                    ylab = "PC2"), ...) 
{
  dp <- dim(phi)
  dt <- dim(theta)
  N <- sum(doc.length)
  W <- length(vocab)
  D <- length(doc.length)
  K <- dt[2]
  if (dp[1] != K) 
    stop("Number of rows of phi does not match \n      number of columns of theta; both should be equal to the number of topics \n      in the model.")
  if (D != dt[1]) 
    stop("Length of doc.length not equal \n      to the number of rows in theta; both should be equal to the number of \n      documents in the data.")
  if (dp[2] != W) 
    stop("Number of terms in vocabulary does \n      not match the number of columns of phi (where each row of phi is a\n      probability distribution of terms for a given topic).")
  if (length(term.frequency) != W) 
    stop("Length of term.frequency \n      not equal to the number of terms in the vocabulary.")
  if (any(nchar(vocab) == 0)) 
    stop("One or more terms in the vocabulary\n      has zero characters -- all terms must have at least one character.")
  phi.test <- all.equal(rowSums(phi), rep(1, K), check.attributes = FALSE)
  theta.test <- all.equal(rowSums(theta), rep(1, dt[1]), check.attributes = FALSE)
  if (!isTRUE(phi.test)) 
    stop("Rows of phi don't all sum to 1.")
  if (!isTRUE(theta.test)) 
    stop("Rows of theta don't all sum to 1.")
  topic.frequency <- colSums(theta * doc.length)
  topic.proportion <- topic.frequency/sum(topic.frequency)
  o <- order(topic.proportion, decreasing = TRUE)
  #phi <- phi[o, ]
  #theta <- theta[, o]
  #topic.frequency <- topic.frequency[o]
  #topic.proportion <- topic.proportion[o]
  mds.res <- mds.method(phi)
  if (is.matrix(mds.res)) {
    colnames(mds.res) <- c("x", "y")
  }
  else if (is.data.frame(mds.res)) {
    names(mds.res) <- c("x", "y")
  }
  else {
    warning("Result of mds.method should be a matrix or data.frame.")
  }
  mds.df <- data.frame(mds.res, topics = seq_len(K), Freq = topic.proportion * 
                         100, cluster = 1, stringsAsFactors = FALSE)
  term.topic.frequency <- phi * topic.frequency
  term.frequency <- colSums(term.topic.frequency)
  stopifnot(all(term.frequency > 0))
  term.proportion <- term.frequency/sum(term.frequency)
  phi <- t(phi)
  topic.given.term <- phi/rowSums(phi)
  kernel <- topic.given.term * log(sweep(topic.given.term, 
                                         MARGIN = 2, topic.proportion, `/`))
  distinctiveness <- rowSums(kernel)
  saliency <- term.proportion * distinctiveness
  default.terms <- vocab[order(saliency, decreasing = TRUE)][1:R]
  counts <- as.integer(term.frequency[match(default.terms, 
                                            vocab)])
  Rs <- rev(seq_len(R))
  default <- data.frame(Term = default.terms, logprob = Rs, 
                        loglift = Rs, Freq = counts, Total = counts, Category = "Default", 
                        stringsAsFactors = FALSE)
  topic_seq <- rep(seq_len(K), each = R)
  category <- paste0("Topic", topic_seq)
  lift <- phi/term.proportion
  find_relevance <- function(i) {
    relevance <- i * log(phi) + (1 - i) * log(lift)
    idx <- apply(relevance, 2, function(x) order(x, decreasing = TRUE)[seq_len(R)])
    indices <- cbind(c(idx), topic_seq)
    data.frame(Term = vocab[idx], Category = category, logprob = round(log(phi[indices]), 
                                                                       4), loglift = round(log(lift[indices]), 4), stringsAsFactors = FALSE)
  }
  lambda.seq <- seq(0, 1, by = lambda.step)
  if (missing(cluster)) {
    tinfo <- lapply(as.list(lambda.seq), find_relevance)
  }
  else {
    tinfo <- parallel::parLapply(cluster, as.list(lambda.seq), 
                                 find_relevance)
  }
  tinfo <- unique(do.call("rbind", tinfo))
  tinfo$Total <- term.frequency[match(tinfo$Term, vocab)]
  rownames(term.topic.frequency) <- paste0("Topic", seq_len(K))
  colnames(term.topic.frequency) <- vocab
  tinfo$Freq <- term.topic.frequency[as.matrix(tinfo[c("Category", 
                                                       "Term")])]
  tinfo <- rbind(default, tinfo)
  ut <- sort(unique(tinfo$Term))
  m <- sort(match(ut, vocab))
  tmp <- term.topic.frequency[, m]
  r <- row(tmp)[tmp >= 0.5]
  c <- col(tmp)[tmp >= 0.5]
  dd <- data.frame(Term = vocab[m][c], Topic = r, Freq = round(tmp[cbind(r, 
                                                                         c)]), stringsAsFactors = FALSE)
  dd[, "Freq"] <- dd[, "Freq"]/term.frequency[match(dd[, 
                                                       "Term"], vocab)]
  token.table <- dd[order(dd[, 1], dd[, 2]), ]
  RJSONIO::toJSON(list(mdsDat = mds.df, tinfo = tinfo, token.table = token.table, 
                       R = R, lambda.step = lambda.step, plot.opts = plot.opts, 
                       topic.order = o))
}

environment(createJSON_noreorder) <- asNamespace('LDAvis')
assignInNamespace('createJSON',createJSON_noreorder,ns='LDAvis')


ggpropovertime <- function(eff_model,topic_model,topics=NULL,npts=100){
  if (is.null(topics)){
    s <- summary(eff_model)$tables
    topics <- na.omit(sapply(seq_along(s), function(i){
      ifelse(any(s[[i]][-1,4] < 0.05),i,NA)
    }))
  }
  
  dat_plot <- plot(eff_model,'year',method='continuous',topics=topics,model=topic_model,npoints=npts)
  names(dat_plot$means) <- paste0('T',dat_plot$topics)
  names(dat_plot$ci) <- paste0('T',dat_plot$topics)

  p1 <- as_tibble(do.call('rbind',lapply(dat_plot$ci,t))) %>% 
    rename('lower'=1,'upper'=2) %>%
    mutate(mean=unlist(dat_plot$means),
           topic=as.factor(rep(dat_plot$topics,each=npts)),
           x=rep(dat_plot$x,length(topics))) %>%
    ggplot(aes(x=x,y=mean,ymin=lower,ymax=upper,color=topic)) +
    geom_pointrange(alpha=0.5)
  print(p1)
}


### loop to generate and clean corpus
files <- file.path('D:/Dropbox/beam/Full_Texts',read_csv('D:/Dropbox/beam/text_titles.csv') %>% select('Text') %>% unlist())
start_words <- c('^abstract|^background|^introduction|abstract$|background$|introduction$')
end_words <- c('^references|^acknowledgements|references$|acknowledgements$')
months <- str_to_lower(month.name)
df <- data_frame()
j <- 1
for (f in files){
  print(j)
  txt <- read_lines(f)
  if (length(txt) < 2) next # skip file if near empty
  
  fn <- str_replace(f,'^D:/Dropbox/beam/Full_Texts/(.*)\\.txt$','\\1')

  i <- 1
  string <- ''
  yr <- NA
  while (i < length(txt)){
    l <- txt[i]
    ll <- str_to_lower(l)
    ll <- str_remove_all(ll,' ')
    
    # look for a published date in beginning of text
    if (is.na(yr)){yr <- str_extract(ll,'^[1-2][9|0][0-9][0-9]$')}
    if (is.na(yr)){
      if (any(str_detect(ll,months))){
        yr <- str_extract(ll,'[1-2][9|0][0-9][0-9]$')
      }
    }
    
    # look for start words (abstract, introduction, etc) to start extraction
    if (str_detect(ll,start_words)){
      i <- i+1
      for (line in txt[i:length(txt)]){
        if (is.na(line)) next
        line <- str_to_lower(line)
        if (str_detect(line,end_words)) break
        line <- str_replace_all(line, '[\r\n]f',' ')
        if (line == '') next
        if (str_detect(line,'issn')) next 
        if (str_detect(line,'$\\[[0-9]+\\]|$\\([0-9]+\\)|^[0-9]+\\.')) next # screen for refs 
        line <- str_remove_all(line,'\\[[0-9]+\\]|\\([0-9]+\\)')
        line <- str_replace_all(line,'[^[:alnum:]]',' ') # remove all non-alphanum
        line <- str_replace_all(line,'Bayesian confidence propagation neural network','BCPNN')
        string <- str_c(string,line)
      }
      break
    }
    i <- i+1
  }
  
  if (string != '') df <- bind_rows(df,data_frame('doc'=fn,'year'=yr,'text'=string))
  
  j <- j+1
}

# saveRDS(df,'D:/Dropbox/beam/beam_tm_df.rds')

df <- readRDS('D:/Dropbox/beam/beam_tm_df.rds') %>%
  left_join(read_csv('D:/Dropbox/beam/text_titles.csv') %>%
              select(doc=Text,year2=`Published Year`) %>%
              mutate(doc=str_remove(doc,'\\.txt$')),
            by='doc') %>%
  mutate(year=ifelse(!is.na(year2),year2,year)) %>%
  select(-year2)

### creating dataframes
df_covs <- df %>% 
  filter(!is.na(year)) %>% 
  mutate(year=as.integer(year)) %>%
  filter(year>2000,year<2022)

df_tidy <- df %>%
  unnest_tokens(word,text) %>%
  anti_join(stop_words) %>%
  filter(!(word %in% as.character(1:100))) 

tfidf <- df_tidy %>%
  count(doc, word, sort = TRUE)


### preprocess corpus
clean_tmp <- textProcessor(documents=df$text,metadata=df,
                       ucp=TRUE,wordLengths=c(2,20),
                       stem=FALSE)
clean_tmp_stemmed <- textProcessor(documents=df$text,metadata=df,
                           ucp=TRUE,wordLengths=c(2,20),
                           stem=TRUE)
clean_covs_tmp <- textProcessor(documents=df_covs$text,metadata=df_covs,
                                ucp=TRUE,wordLengths=c(2,20),
                                stem=FALSE)
clean_covs_tmp_stemmed <- textProcessor(documents=df_covs$text,metadata=df_covs,
                                ucp=TRUE,wordLengths=c(2,20),
                                stem=TRUE)

# png('D:/Dropbox/beam/filter_thres.png',width=1000,height=400)
# plotRemoved(clean_tmp$documents,lower.thresh=seq(1,50,by=10))
# dev.off()

clean <- prepDocuments(clean_tmp$documents,clean_tmp$vocab,
                       clean_tmp$meta,lower.thresh=10)

clean_covs <- prepDocuments(clean_covs_tmp$documents,clean_covs_tmp$vocab,
                            clean_covs_tmp$meta,lower.thresh=10)

clean_stemmed <- prepDocuments(clean_tmp_stemmed$documents,clean_tmp_stemmed$vocab,
                       clean_tmp_stemmed$meta,lower.thresh=10)

clean_covs_stemmed <- prepDocuments(clean_covs_tmp_stemmed$documents,clean_covs_tmp_stemmed$vocab,
                                    clean_covs_tmp_stemmed$meta,lower.thresh=10)


### search for optimal K
# kres <- searchK(clean$documents,clean$vocab,K=seq(5,45,5),data=clean$meta,heldout.seed=1123)
# kres_covs <- searchK(clean_covs$documents,clean_covs$vocab,K=seq(5,30,5),
#                      prevalence=~year,data=clean_covs$meta,heldout.seed=1123)
# kres_covs_spline <- searchK(clean_covs$documents,clean_covs$vocab,K=seq(5,30,5),
#                             prevalence=~s(year,df=3),data=clean_covs$meta,heldout.seed=1123)
# kres_stemmed <- searchK(clean_stemmed$documents,clean_stemmed$vocab,K=seq(5,30,5),data=clean_stemmed$meta,heldout.seed=1123)
# kres_covs_stemmed <- searchK(clean_covs_stemmed$documents,clean_covs_stemmed$vocab,K=seq(5,30,5),
#                      prevalence=~year,data=clean_covs_stemmed$meta,heldout.seed=1123)
# kres_covs_spline_stemmed <- searchK(clean_covs_stemmed$documents,clean_covs_stemmed$vocab,K=seq(5,30,5),
#                             prevalence=~s(year,df=3),data=clean_covs_stemmed$meta,heldout.seed=1123)
# saveRDS(list(kres=kres,kres_covs=kres_covs,kres_covs_spline=kres_covs_spline,
#              kres_stemmed=kres_stemmed,kres_covs_stemmed=kres_covs_stemmed,
#              kres_covs_spline_stemmed=kres_covs_spline_stemmed),'D:/Dropbox/beam/beam_kres.rds')

ksearch <- readRDS('D:/Dropbox/beam/beam_kres.rds')
ksearch$kres #30
ksearch$kres_covs #25
ksearch$kres_covs_spline #25
ksearch$kres_stemmed #25
ksearch$kres_covs_spline_stemmed #25


### create topic models
# # corr topic model 30 topics
# tm <- stm(documents=clean$documents,vocab=clean$vocab,
#           K=30,verbose=TRUE,init.type='Spectral',seed=123)
# 
# # corr topic model with 25 topics w/ topic props regressed on year
# tm_cov <- stm(documents=clean_covs_stemmed$documents,vocab=clean_covs_stemmed$vocab,
#               K=25,prevalence=~year,data=clean_covs_stemmed$meta,
#               verbose=TRUE,init.type='Spectral',seed=123)
# 
# # corr topic model with 25 topics w/ topic props regressed/spline on year
# tm_cov_spline <- stm(documents=clean_covs_stemmed$documents,vocab=clean_covs_stemmed$vocab,
#               K=25,prevalence=~s(year,df=3),data=clean_covs_stemmed$meta,
#               verbose=TRUE,init.type='Spectral',seed=123)
# 
# # corr topic model 25 topics
# tm_stemmed <- stm(documents=clean_stemmed$documents,vocab=clean_stemmed$vocab,
#           K=25,verbose=TRUE,init.type='Spectral',seed=123)
# 
# # corr topic model with 25 topics w/ topic props regressed on year
# tm_cov_stemmed <- stm(documents=clean_covs_stemmed$documents,vocab=clean_covs_stemmed$vocab,
#               K=25,prevalence=~year,data=clean_covs_stemmed$meta,
#               verbose=TRUE,init.type='Spectral',seed=123)
# 
# # corr topic model with 25 topics w/ topic props regressed/spine on year
# tm_cov_spline_stemmed <- stm(documents=clean_covs_stemmed$documents,vocab=clean_covs_stemmed$vocab,
#                       K=25,prevalence=~s(year,df=3),data=clean_covs_stemmed$meta,
#                       verbose=TRUE,init.type='Spectral',seed=123)
# 
# 
# saveRDS(list(tm=tm,tm_cov=tm_cov,tm_cov_spline=tm_cov_spline,
#              tm_stemmed=tm_stemmed,tm_cov_stemmed=tm_cov_stemmed,
#              tm_cov_spline_stemmed=tm_cov_spline_stemmed,
#              clean=clean,clean_covs=clean_covs,
#              clean_stemmed=clean_stemmed,clean_covs_stemmed=clean_covs_stemmed),
#         'D:/Dropbox/beam/beam_tm.rds')


dat <- readRDS('D:/Dropbox/beam/beam_tm.rds')
tm <- dat$tm
tm_cov <- dat$tm_cov
tm_cov_spline <- dat$tm_cov_spline
tm_stemmed <- dat$tm_stemmed
tm_cov_stemmed <- dat$tm_cov_stemmed
tm_cov_spline_stemmed <- dat$tm_cov_spline_stemmed
clean <- dat$clean
clean_covs <- dat$clean_covs
clean_stemmed <- dat$clean_stemmed
clean_covs_stemmed <- dat$clean_covs_stemmed



### LDAvis
toLDAvis(tm,docs=clean$documents,out.dir='D:/Dropbox/beam/tm')
toLDAvis(tm_cov,docs=clean_covs$documents,out.dir='D:/Dropbox/beam/tm_cov')
toLDAvis(tm_cov_spline,docs=clean_covs$documents,out.dir='D:/Dropbox/beam/tm_cov_spline',reorder.topics=FALSE,)
toLDAvis(tm_stemmed,docs=clean_stemmed$documents,out.dir='D:/Dropbox/beam/tm_stemmed')
toLDAvis(tm_cov_stemmed,docs=clean_covs_stemmed$documents,out.dir='D:/Dropbox/beam/tm_cov_stemmed')
toLDAvis(tm_cov_spline_stemmed,docs=clean_covs_stemmed$documents,out.dir='D:/Dropbox/beam/tm_cov_spline_stemmed')


### generate topic effect from year
set.seed(123412)
eff <- estimateEffect(1:25 ~ year,tm_cov,meta=clean_covs$meta,
                      uncertainty='Global')
summary(eff)

eff_spline <- estimateEffect(1:25 ~ s(year,df=3),tm_cov_spline,meta=clean_covs$meta,
                             uncertainty='Global')
summary(eff_spline)

eff_stemmed <- estimateEffect(1:25 ~ year,tm_cov_stemmed,meta=clean_covs_stemmed$meta,
                      uncertainty='Global')
summary(eff_stemmed)

eff_spline_stemmed <- estimateEffect(1:25 ~ s(year,df=3),tm_cov_spline_stemmed,meta=clean_covs_stemmed$meta,
                              uncertainty='Global')
summary(eff_spline_stemmed)



### plot effects: topic prop over time
ggpropovertime(eff,tm_cov)
ggpropovertime(eff_spline,tm_cov)
ggpropovertime(eff_stemmed,tm_cov_stemmed)
ggpropovertime(eff_spline_stemmed,tm_cov_spline_stemmed)

### some nice summary data
plot(tm,type='summary',n=10)
plot(tm_cov,type='summary',n=10)
plot(tm_cov_spline,type='summary',n=10)
plot(tm_stemmed,type='summary',n=10)
plot(tm_cov_stemmed,type='summary',n=10)
plot(tm_cov_spline_stemmed,type='summary',n=10)


### topic correlations
tm_corr <- topicCorr(tm)
tm_cov_corr <- topicCorr(tm_cov)
tm_cov_spline_corr <- topicCorr(tm_cov_spline)
tm_corr_stemmed <- topicCorr(tm_stemmed)
tm_cov_corr_stemmed <- topicCorr(tm_cov_stemmed)
tm_many_corr_spline_stemmed <- topicCorr(tm_cov_spline_stemmed)


### plot topic correlations
plot(tm_corr)
plot(tm_cov_corr)
plot(tm_cov_spline_corr)
plot(tm_corr_stemmed)
plot(tm_cov_corr_stemmed)
plot(tm_many_corr_spline_stemmed)
