#!/bin/bash
# Dataset description: http://corpus-texmex.irisa.fr
if [ ! -d siftsmall ]; then
  echo 'Downloading siftsmall'
  wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
  tar xzf siftsmall.tar.gz
  cd siftsmall
  prename 's/siftsmall_//' *
  cd ..
fi
if [ ! -d sift ]; then
  echo 'Downloading sift'
  #wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
  tar xzf sift.tar.gz
  cd sift
  prename 's/sift_//' *
  cd ..
fi
