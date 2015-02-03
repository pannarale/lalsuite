#!/bin/sh

# Script to run PulsarTOATest on some simulated pulsar time of arrivals created with
# TEMPO2 based on the par file TOAtest.par (which contains a set of parameters for the
# pulsar J1909-3744). This has been generated by TEMPO2 with the command (to have 1
# observation per day for 1000 days):
# tempo2 -gr fake -f TOAtest.par -ndobs 1 -nobsd 1 -start 54500 -end 55500 -ha 8 -randha n -rms 0 -format tempo2
#
# Note that the PulsarTOATest code needs the pks2gps.clk conversion file to convert
# Parkes observatory times to GPS times. This file has been taken from the TEMPO2
# package tempo2/T2runtime/clock/pks2gps.clk.

if [ -z "${srcdir}" ]; then
    srcdir=`dirname $0`
fi

builddir="./"
CODENAME=${builddir}/PulsarTOATest

$CODENAME -p ${srcdir}/TOAtest.par -t ${srcdir}/TOAtest.simulate -s -c ${srcdir}/pks2gps.clk

if [ $? != "0" ]; then
  exit 1
fi

exit 0