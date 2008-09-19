/*
 * $Id$
 *
 * Copyright (C) 2007  Kipp Cannon and Flanagan, E.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 2 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


/*
 * ============================================================================
 *
 *                                  Preamble
 *
 * ============================================================================
 */


#include <math.h>


#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>


#include <lal/Date.h>
#include <lal/LALComplex.h>
#include <lal/LALDatatypes.h>
#include <lal/LIGOMetadataTables.h>
#include <lal/LIGOMetadataUtils.h>
#include <lal/RealFFT.h>
#include <lal/Sequence.h>
#include <lal/TFTransform.h>
#include <lal/Thresholds.h>
#include <lal/XLALError.h>


#include <lal/LALRCSID.h>
NRCSID(FREQSERIESTOTFPLANEC, "$Id:");


/*
 * ============================================================================
 *
 *                      Time-Frequency Plane Projection
 *
 * ============================================================================
 */


/*
 * Multiply the data by the filter.  The check that the frequency
 * resolutions and units are compatible is omitted because it is implied by
 * the calling code.
 */


static double min(double a, double b)
{
	return a < b ? a : b;
}


static double max(double a, double b)
{
	return a > b ? a : b;
}


static COMPLEX16Sequence *apply_filter(
	COMPLEX16Sequence *outputseq,
	const COMPLEX16FrequencySeries *inputseries,
	const COMPLEX16FrequencySeries *filterseries
)
{
	static const char func[] = "apply_filter";
	/* find bounds of common frequencies */
	const double flo = max(filterseries->f0, inputseries->f0);
	const double fhi = min(filterseries->f0 + filterseries->data->length * filterseries->deltaF, inputseries->f0 + inputseries->data->length * inputseries->deltaF);
	COMPLEX16 *output = outputseq->data + (int) floor((flo - inputseries->f0) / inputseries->deltaF + 0.5);
	COMPLEX16 *last = outputseq->data + (int) floor((fhi - inputseries->f0) / inputseries->deltaF + 0.5);
	const COMPLEX16 *input = inputseries->data->data + (int) floor((flo - inputseries->f0) / inputseries->deltaF + 0.5);
	const COMPLEX16 *filter = filterseries->data->data + (int) floor((flo - filterseries->f0) / filterseries->deltaF + 0.5);

	if(outputseq->length != inputseries->data->length)
		XLAL_ERROR_NULL(func, XLAL_EBADLEN);

	if(((unsigned) (output - outputseq->data) > outputseq->length) || (last - outputseq->data < 0))
		/* inputseries and filterseries don't intersect */
		memset(outputseq->data, 0, outputseq->length * sizeof(*outputseq->data));
	else {
		/* output = inputseries * conj(filter) */
		memset(outputseq->data, 0, (output - outputseq->data) * sizeof(*outputseq->data));
		for(; output < last; output++, input++, filter++)
			*output = XLALCOMPLEX16Mul(*input, XLALCOMPLEX16Conjugate(*filter));
		memset(last, 0, (outputseq->length - (last - outputseq->data)) * sizeof(*outputseq->data));
	}

	return outputseq;
}


/*
 * Project a frequency series onto the comb of channel filters
 */


int XLALFreqSeriesToTFPlane(
	REAL8TimeFrequencyPlane *plane,
	const LALExcessPowerFilterBank *filter_bank,
	const COMPLEX16FrequencySeries *fseries,
	const REAL8FFTPlan *reverseplan
)
{
	static const char func[] = "XLALFreqSeriesToTFPlane";
	COMPLEX16Sequence *fcorr;
	unsigned i;

	/* check input parameters */
	if((fmod(plane->deltaF, fseries->deltaF) != 0.0) ||
	   (fmod(plane->flow - fseries->f0, fseries->deltaF) != 0.0))
		XLAL_ERROR(func, XLAL_EINVAL);

	/* make sure the frequency series spans an appropriate band */
	if((plane->flow < fseries->f0) ||
	   (plane->flow + plane->channel_data->size2 * plane->deltaF > fseries->f0 + fseries->data->length * fseries->deltaF))
		XLAL_ERROR(func, XLAL_EDATA);

	/* create temporary vectors */
	fcorr = XLALCreateCOMPLEX16Sequence(fseries->data->length);
	if(!fcorr)
		XLAL_ERROR(func, XLAL_EFUNC);

#if 0
	/* diagnostic code to dump data for the \hat{s}_{k} histogram */
	{
	unsigned k;
	FILE *f = fopen("sk.dat", "a");
	for(k = plane->flow / fseries->deltaF; k < (plane->flow + plane->channel_data->size2 * plane->deltaF) / fseries->deltaF; k++)
		fprintf(f, "%g\n%g\n", fseries->data->data[k].re, fseries->data->data[k].im);
	fclose(f);
	}
#endif
#if 0
	/* diagnostic code to dump data for the \hat{s}_{k}
	 * \hat{s}^{*}_{k'} histogram */
	{
	unsigned k, dk;
	FILE *f = fopen("sksk.dat", "a");
	for(dk = 0; dk < 100; dk++) {
		double avg_r = 0;
		double avg_i = 0;
	for(k = plane->flow / fseries->deltaF; k + dk < (plane->flow + plane->channel_data->size2 * plane->deltaF) / fseries->deltaF; k++) {
		double dr = fseries->data->data[k].re;
		double di = fseries->data->data[k].im;
		double dkr = fseries->data->data[k + dk].re;
		double dki = fseries->data->data[k + dk].im;
		avg_r += dr * dkr + di * dki;
		avg_i += di * dkr - dr * dki;
	}
		avg_r /= k - plane->flow / fseries->deltaF;
		avg_i /= k - plane->flow / fseries->deltaF;
		fprintf(f, "%d %g %g\n", dk, avg_r, avg_i);
	}
	fclose(f);
	}
#endif

	/* loop over the time-frequency plane's channels */
	for(i = 0; i < plane->channel_data->size2; i++) {
		unsigned j;
		/* cross correlate the input data against the channel
		 * filter by taking their product in the frequency domain
		 * and then inverse transforming to the time domain to
		 * obtain an SNR time series.  Note that
		 * XLALREAL8ReverseFFT() omits the factor of 1 / (N Delta
		 * t) in the inverse transform. */
		apply_filter(fcorr, fseries, filter_bank->basis_filters[i].fseries);
		if(XLALREAL8ReverseFFT(plane->channel_buffer, fcorr, reverseplan)) {
			XLALDestroyCOMPLEX16Sequence(fcorr);
			XLAL_ERROR(func, XLAL_EFUNC);
		}
		/* interleave the result into the channel_data array */
		for(j = 0; j < plane->channel_buffer->length; j++)
			gsl_matrix_set(plane->channel_data, j, i, plane->channel_buffer->data[j]);
	}

	/* clean up */
	XLALDestroyCOMPLEX16Sequence(fcorr);

	/* set the name and epoch of the TF plane */
	strncpy(plane->name, fseries->name, LALNameLength);
	plane->epoch = fseries->epoch;

	/* success */
	return 0;
}


/*
 * ============================================================================
 *
 *                                   Output
 *
 * ============================================================================
 */


/*
 * Convert time-frequency tile info to a SnglBurst row.
 */


static SnglBurst *XLALTFTileToBurstEvent(
	const REAL8TimeFrequencyPlane *plane,
	unsigned tile_start,
	unsigned tile_length,
	double f_centre,
	double bandwidth,
	double h_rss,
	double E,
	double d,
	double confidence
)
{
	static const char func[] = "XLALTFTileToBurstEvent";
	SnglBurst *event = XLALCreateSnglBurst();

	if(!event)
		XLAL_ERROR_NULL(func, XLAL_ENOMEM);

	event->next = NULL;
	strncpy(event->ifo, plane->name, 2);
	event->ifo[2] = '\0';
	strncpy(event->search, "excesspower", LIGOMETA_SEARCH_MAX);
	event->search[LIGOMETA_SEARCH_MAX - 1] = '\0';
	strncpy(event->channel, plane->name, LIGOMETA_CHANNEL_MAX);
	event->channel[LIGOMETA_CHANNEL_MAX - 1] = '\0';
	event->start_time = plane->epoch; 
	XLALGPSAdd(&event->start_time, tile_start * plane->deltaT);
	event->duration = tile_length * plane->deltaT;
	event->peak_time = event->start_time;
	XLALGPSAdd(&event->peak_time, event->duration / 2);
	event->bandwidth = bandwidth;
	event->central_freq = f_centre;
	/* FIXME: put h_rss into the "hrss" column */
	event->amplitude = h_rss;
	event->snr = E / d - 1;
	/* -ln P(event | stationary Gaussian white noise) */
	event->confidence = confidence;

	return event;
}


/*
 * ============================================================================
 *
 *                               Tile Analysis
 *
 * ============================================================================
 */


static double compute_unwhitened_mean_square(
	const LALExcessPowerFilterBank *filter_bank,
	unsigned channel,
	unsigned channels
)
{
	unsigned i;
	double mean_square = 0;

	for(i = channel; i < channel + channels; i++)
		mean_square += pow(filter_bank->basis_filters[i].unwhitened_rms, 2);

	return mean_square;
}


SnglBurst *XLALComputeExcessPower(
	const REAL8TimeFrequencyPlane *plane,
	const LALExcessPowerFilterBank *filter_bank,
	SnglBurst *head,
	double confidence_threshold
)
{
	static const char func[] = "XLALComputeExcessPower";
	gsl_vector filter_output;
	gsl_vector channel_buffer;
	gsl_vector unwhitened_channel_buffer;
	unsigned start;
	unsigned end;
	unsigned channel;
	unsigned channels;
	unsigned channel_end;
	double h_rss;
	double confidence;
	/* number of degrees of freedom in tile = number of
	 * "virtual pixels" in tile. */
	double tile_dof;

	/* argh!  C90 ... */
	filter_output.size = plane->channel_data->size1;
	filter_output.stride = plane->channel_data->size2;
	filter_output.data = NULL;
	filter_output.block = NULL;
	filter_output.owner = 0;
	channel_buffer.size = plane->channel_buffer->length;
	channel_buffer.stride = 1;
	channel_buffer.data = plane->channel_buffer->data;
	channel_buffer.block = NULL;
	channel_buffer.owner = 0;
	unwhitened_channel_buffer.size = plane->unwhitened_channel_buffer->length;
	unwhitened_channel_buffer.stride = 1;
	unwhitened_channel_buffer.data = plane->unwhitened_channel_buffer->data;
	unwhitened_channel_buffer.block = NULL;
	unwhitened_channel_buffer.owner = 0;

	for(channels = plane->tiles.min_channels; channels <= plane->tiles.max_channels; channels *= 2) {
		/* compute distance between "virtual pixels" for this
		 * (wide) channel */
		const unsigned stride = floor(1.0 / (channels * plane->tiles.dof_per_pixel) + 0.5);

	for(channel_end = (channel = 0) + channels; channel_end <= plane->channel_data->size2; channel_end = (channel += channels / plane->tiles.inv_fractional_stride) + channels) {
		/* the root mean square of the "virtual channel",
		 * \sqrt{\mu^{2}} in the algorithm description */
		const double sample_rms = sqrt(channels * plane->deltaF / plane->fseries_deltaF + XLALREAL8SequenceSum(filter_bank->twice_channel_overlap, channel, channels - 1));
		/* the root mean square of the "uwapprox" quantity computed
		 * below, which is proportional to an approximation of the
		 * unwhitened time series. */
		double uwsample_rms;
		/* true unwhitened root mean square for this channel.  the
		 * ratio of this squared to uwsample_rms^2 is the
		 * correction factor to be applied to uwapprox^2 to convert
		 * it to an approximation of the square of the unwhitened
		 * channel */
		const double strain_rms = sqrt(compute_unwhitened_mean_square(filter_bank, channel, channels) + XLALREAL8SequenceSum(filter_bank->unwhitened_cross, channel, channels - 1));
		unsigned c;
		unsigned t;

		/* compute uwsample_rms */
		uwsample_rms = compute_unwhitened_mean_square(filter_bank, channel, channels);
		for(c = channel; c < channel_end - 1; c++)
			uwsample_rms += filter_bank->twice_channel_overlap->data[c] * filter_bank->basis_filters[c].unwhitened_rms * filter_bank->basis_filters[c + 1].unwhitened_rms * plane->fseries_deltaF / plane->deltaF;
		uwsample_rms = sqrt(uwsample_rms);

		/* reconstruct the time series and unwhitened time series
		 * for this (possibly multi-filter) channel.  both time
		 * series are normalized so that each sample has a mean
		 * square of 1 */
		filter_output.data = plane->channel_data->data + channel;
		gsl_vector_set_zero(&channel_buffer);
		gsl_vector_set_zero(&unwhitened_channel_buffer);
		for(c = channel; c < channel_end; filter_output.data++, c++) {
			gsl_blas_daxpy(1.0 / sample_rms, &filter_output, &channel_buffer);
			gsl_blas_daxpy(filter_bank->basis_filters[c].unwhitened_rms * sqrt(plane->fseries_deltaF / plane->deltaF) / uwsample_rms, &filter_output, &unwhitened_channel_buffer);
		}

#if 0
		/* diagnostic code to dump data for the s_{j} histogram */
		{
		FILE *f = fopen("sj.dat", "a");
		for(t = 0; t < plane->channel_buffer->length; t++)
			fprintf(f, "%g\n", plane->unwhitened_channel_buffer->data[t]);
		fclose(f);
		}
#endif

		/* square the samples in the channel time series because
		 * from now on that's all we'll need */
		for(t = 0; t < channel_buffer.size; t++) {
			gsl_vector_set(&channel_buffer, t, pow(gsl_vector_get(&channel_buffer, t), 2));
			gsl_vector_set(&unwhitened_channel_buffer, t, pow(gsl_vector_get(&unwhitened_channel_buffer, t), 2));
		}

	/* start with at least 2 degrees of freedom */
	for(tile_dof = 2; tile_dof <= plane->tiles.max_length / stride; tile_dof *= 2) {
		unsigned length = tile_dof * stride;

	for(end = (start = plane->tiles.tiling_start) + length; end <= plane->tiles.tiling_end; end = (start += length / plane->tiles.inv_fractional_stride) + length) {
		double sumsquares = 0;
		double uwsumsquares = 0;

		/* compute sum of squares, and unwhitened sum of squares
		 * (samples have already been squared) */
		for(t = start + stride / 2; t < end; t += stride) {
			sumsquares += gsl_vector_get(&channel_buffer, t);
			uwsumsquares += gsl_vector_get(&unwhitened_channel_buffer, t);
		}

		/* compute statistical confidence */
		/* FIXME:  the 0.62 is an empirically determined
		 * degree-of-freedom fudge factor.  figure out what its
		 * origin is, and account for it correctly.  it's most
		 * likely due to the time-frequency plane pixels not being
		 * independent of one another as a consequence of a
		 * non-zero inner product of the time-domain impulse
		 * response of the channel filter for adjacent pixels */
		confidence = -XLALlnOneMinusChisqCdf(sumsquares * .62, tile_dof * .62);
		if(XLALIsREAL8FailNaN(confidence))
			XLAL_ERROR_NULL(func, XLAL_EFUNC);

		/* record tiles whose statistical confidence is above
		 * threshold and that have real-valued h_rss */
		if((confidence >= confidence_threshold) && (uwsumsquares >= tile_dof)) {
			SnglBurst *oldhead = head;

			/* compute h_rss */
			h_rss = sqrt((uwsumsquares - tile_dof) * (stride * plane->deltaT)) * strain_rms;

			/* add new event to head of linked list */
			head = XLALTFTileToBurstEvent(plane, start, length, plane->flow + (channel + .5 * channels) * plane->deltaF, channels * plane->deltaF, h_rss, sumsquares, tile_dof, confidence);
			if(!head)
				XLAL_ERROR_NULL(func, XLAL_EFUNC);
			head->next = oldhead;
		}
	}
	}
	}
	}

	/* success */
	return(head);
}
