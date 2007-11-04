# $Id$
#
# Copyright (C) 2007  Kipp C. Cannon
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import math
import numpy
from scipy.interpolate import interpolate
import sys


from glue.ligolw import lsctables


__author__ = "Kipp Cannon <kipp@gravity.phys.uwm.edu>"
__version__ = "$Revision$"[11:-2]
__date__ = "$Date$"[7:-2]


#
# =============================================================================
#
#                                  Likelihood
#
# =============================================================================
#


#
# starting from Bayes' theorem:
#
# P(coinc is a g.w. | its parameters)
#     P(those parameters | a coinc known to be a g.w.) * P(coinc is g.w.)
#   = -------------------------------------------------------------------
#                                P(parameters)
#
#     P(those parameters | a coinc known to be a g.w.) * P(coinc is g.w.)
#   = -------------------------------------------------------------------
#     P(noise params) * P(coinc is not g.w.) + P(inj params) * P(coinc is g.w.)
#
#                       P(inj params) * P(coinc is g.w.)
#   = -------------------------------------------------------------------
#     P(noise params) * [1 - P(coinc is g.w.)] + P(inj params) * P(coinc is g.w.)
#
#                        P(inj params) * P(coinc is g.w.)
#   = ----------------------------------------------------------------------
#     P(noise params) + [P(inj params) - P(noise params)] * P(coinc is g.w.)
#


#
# How to make an interpolator
#


def make_interp(x, y):
	# extrapolate x and y arrays by one element at each end.  this has
	# to be done because the Rate class in pylal.rate returns the x
	# co-ordinates as the bin centres, which is correct, but it means
	# that an event can have a set of parameter values that lie beyond
	# the end of the x co-ordinate array (the parameters are still in
	# the bin, but in the outer half), meanwhile the scipy interpolator
	# insists on only interpolating (go figger) so it throws an error
	# when such an event is encountered.  Note that because the y
	# values are assumed to be probabilities they cannot be negative
	# but that constraint is not imposed here so it must be imposed
	# elsewhere.

	x = numpy.hstack((x[0] + (x[0] - x[1]), x, x[-1] + (x[-1] - x[-2])))
	y = numpy.hstack((y[0] + (y[0] - y[1]), y, y[-1] + (y[-1] - y[-2])))

	return interpolate.interp1d(x, y)


#
# Class for computing foreground likelihoods from the measurements in a
# CoincParamsDistributions instance.
#


class Likelihood(object):
	def __init__(self, coinc_param_distributions):
		# construct interpolators from the distribution data
		self.background_rates = {}
		self.injection_rates = {}
		for name, rate in coinc_param_distributions.background_rates.iteritems():
			self.background_rates[name] = make_interp(rate.centres()[0], rate.array)
		for name, rate in coinc_param_distributions.injection_rates.iteritems():
			self.injection_rates[name] = make_interp(rate.centres()[0], rate.array)

	def set_P_gw(self, P):
		self.P_gw = P

	def P(self, param_func, events, offsetdict):
		P_background = 1.0
		P_injection = 1.0
		for name, value in param_func(events, offsetdict).iteritems():
			try:
				# the interpolators might return negative
				# values for the probabilities.  the
				# constraint that probabilities be
				# non-negative is imposed here.
				P_b = max(0, self.background_rates[name](value)[0])
				P_i = max(0, self.injection_rates[name](value)[0])
			except ValueError:
				# param value is outside an interpolator
				# domain, so skip on the reasoning that
				# this parameter provides no information
				continue
			P_background *= P_b
			P_injection *= P_i
		return P_background, P_injection

	def __call__(self, param_func, events, offsetdict):
		"""
		Compute the likelihood that the coincident n-tuple of
		events are the result of a gravitational wave:  the
		probability that the hypothesis "the events are a
		gravitational wave" is correct, in the context of the
		measured background and foreground distributions, and the
		intrinsic rate of gravitational wave coincidences.  offsets
		is a dictionary of instrument --> offset mappings to be
		used to time shift the events before comparison.
		"""
		P_background, P_injection = self.P(param_func, events, offsetdict)
		return (P_injection * self.P_gw) / (P_background + (P_injection - P_background) * self.P_gw)


class Confidence(Likelihood):
	def __call__(self, param_func, events, offsetdict):
		"""
		Compute the confidence that the list of events are the
		result of a gravitational wave:  -ln[1 - P(gw)], where
		P(gw) is the likelihood returned by the Likelihood class.
		A set of events very much like gravitational waves will
		have a likelihood of being a gravitational wave very close
		to 1, so 1 - P is a small positive number, and so -ln of
		that is a large positive number.
		"""
		P_bak, P_inj = self.P(param_func, events, offsetdict)
		return  math.log(P_bak + (P_inj - P_bak) * self.P_gw) - math.log(P_inj) - math.log(self.P_gw)


class LikelihoodRatio(Likelihood):
	def set_P_gw(self, P):
		"""
		Raises NotImplementedError.  The likelihood ratio is
		computed without using this parameter.
		"""
		raise NotImplementedError

	def __call__(self, param_func, events, offsetdict):
		"""
		Compute the likelihood ratio for the hypothesis that the
		list of events are the result of a gravitational wave.  The
		likelihood ratio is the ratio P(inj params) / P(noise
		params).  The probability that the events are the result of
		a gravitiational wave is a monotonically increasing
		function of the likelihood ratio, so ranking events from
		"most like a gravitational wave" to "least like a
		gravitational wave" can be performed by calculating the
		likelihood ratios, which has the advantage of not requiring
		a prior probability to be provided.
		"""
		P_bak, P_inj = self.P(param_func, events, offsetdict)
		if P_bak == 0 and P_inj == 0:
			# this can happen.  "correct" answer is 0, not NaN,
			# because if a tuple of events has been found in a
			# region of parameter space where the probability
			# of an injection occuring is 0 then there is no
			# way this is an injection.  there is also,
			# aparently, no way it's a noise event, but that's
			# irrelevant because we are supposed to be
			# computing something that is a monotonically
			# increasing function of the probability that an
			# event tuple is a gravitational wave, which is 0
			# in this part of the parameter space.
			return 0
		return  P_inj / P_bak


#
# =============================================================================
#
#                              Library Interface
#
# =============================================================================
#


#
# Main routine
#


def ligolw_burca2(database, likelihood_ratio, coinc_params, verbose = False):
	"""
	Assigns likelihood ratio values to excess power coincidences.
	database is pylal.SnglBurstUtils.CoincDatabase instance, and
	likelihood_ratio is a LikelihoodRatio class instance.
	"""
	#
	# Find document parts.
	#

	if None in (database.coinc_def_table, database.coinc_table, database.time_slide_table):
		raise ValueError, "database appears to be missing coinc tables"

	definer_ids = set([database.coinc_def_table.get_coinc_def_id([lsctables.SnglBurstTable.tableName], create_new = False)])
	try:
		definer_ids.add(database.coinc_def_table.get_coinc_def_id([lsctables.SnglBurstTable.tableName, lsctables.SimBurstTable.tableName], create_new = False))
	except KeyError:
		# there appear to be no injections in this file
		pass
	time_slides = database.time_slide_table.as_dict()

	#
	# Iterate over all coincs, assigning likelihood ratios to
	# burst+burst coincs, and sim+burst coincs if the document contains
	# them.
	#

	if verbose:
		print >>sys.stderr, "computing likelihood ratios ..."
		n_coincs = len(database.coinc_table)

	cursor = database.connection.cursor()
	for n, (coinc_event_id, time_slide_id) in enumerate(database.connection.cursor().execute("SELECT coinc_event_id, time_slide_id FROM coinc_event WHERE coinc_def_id IN (%s)" % ", ".join(["\"%s\"" % str(id) for id in definer_ids]))):
		if verbose and not n % 200:
			print >>sys.stderr, "\t%.1f%%\r" % (100.0 * n / n_coincs),

		# retrieve sngl_burst events, sorted by instrument
		# name
		events = map(database.sngl_burst_table._row_from_cols, cursor.execute("""
SELECT
	sngl_burst.*
FROM
	sngl_burst
	JOIN coinc_event_map ON (
		coinc_event_map.table_name == 'sngl_burst'
		AND sngl_burst.event_id == coinc_event_map.event_id
	)
WHERE
	coinc_event_map.coinc_event_id == ?
ORDER BY
	sngl_burst.ifo
		""", (coinc_event_id,)))

		# compute likelihood ratio
		cursor.execute("""
UPDATE
	coinc_event
SET
	likelihood = ?
WHERE
	coinc_event_id == ?
		""", (likelihood_ratio(coinc_params, events, time_slides[time_slide_id]), coinc_event_id))
	if verbose:
		print >>sys.stderr, "\t100.0%"

	#
	# Done
	#

	return database
