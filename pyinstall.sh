#!/bin/bash

reset && sudo rm -rf /usr/local/lib/python2.7/dist-packages/*{_{mean,k},vq}tree* build && sudo python setup.py install
