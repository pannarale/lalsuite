/*
 *
 * Copyright (C) 2007  Kipp Cannon and Flanagan, E
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


#include <complex.h>
#include <math.h>


#include <gsl/gsl_matrix.h>


#include <lal/Date.h>
#include <lal/FrequencySeries.h>
#include <lal/LALAtomicDatatypes.h>
#include <lal/LALMalloc.h>
#include <lal/LALStdlib.h>
#include <lal/RealFFT.h>
#include <lal/Sequence.h>
#include <lal/TFTransform.h>
#include <lal/TimeFreqFFT.h>
#include <lal/Units.h>
#include <lal/Window.h>
#include <lal/XLALError.h>

static double min(double a, double b) { return a < b ? a : b; }
static double max(double a, double b) { return a > b ? a : b; }


/*
 * ============================================================================
 *
 *                             Timing Arithmetic
 *
 * ============================================================================
 */


/**
 * Round target_length down so that an integer number of intervals of
 * length segment_length, each shifted by segment_shift with respect to the
 * interval preceding it, fits into the result.
 */
INT4 XLALOverlappedSegmentsCommensurate(
	INT4 target_length,
	INT4 segment_length,
	INT4 segment_shift
)
{
	UINT4 segments;

	/*
	 * check input
	 */

	if(segment_length < 1) {
		XLALPrintError("segment_length < 1");
		XLAL_ERROR(XLAL_EINVAL);
	}
	if(segment_shift < 1) {
		XLALPrintError("segment_shift < 1");
		XLAL_ERROR(XLAL_EINVAL);
	}

	/*
	 * trivial case
	 */

	if(target_length < segment_length)
		return 0;

	/*
	 * do the arithmetic
	 */

	segments = (target_length - segment_length) / segment_shift;

	return segments * segment_shift + segment_length;
}


/**
 * Compute and return the timing parameters for an excess power analysis.
 * Pass NULL for any optional pointer to not compute and return that
 * parameter.  The return is 0 on success, negative on failure.
 */
int XLALEPGetTimingParameters(
	int window_length,	/**< Number of samples in a window used for the time-frequency plane */
	int max_tile_length,	/**< Number of samples in the tile of longest duration */
	double fractional_tile_shift,	/**< Number of samples by which the start of the longest tile is shifted from the start of the tile preceding it, as a fraction of its length */
	int *psd_length,	/**< (optional) User's desired number of samples to use in computing a PSD estimate.  Will be replaced with actual number of samples to use in computing a PSD estimate (rounded down to be comensurate with the windowing). */
	int *psd_shift,	/**< (optional) Number of samples by which the start of a PSD is to be shifted from the start of the PSD that preceded it in order that the tiling pattern continue smoothly across the boundary. */
	int *window_shift,	/**< Number of samples by which the start of a time-frequency plane window is shifted from the window preceding it in order that the tiling pattern continue smoothly across the boundary. */
	int *window_pad,	/**< How many samples at the start and end of each window are treated as padding, and will not be covered by the tiling. */
	int *tiling_length	/**< How many samples will be covered by the tiling. */
)
{
	int max_tile_shift = fractional_tile_shift * max_tile_length;

	/*
	 * check input parameters
	 */

	if(window_length % 4 != 0) {
		XLALPrintError("window_length is not a multiple of 4");
		XLAL_ERROR(XLAL_EINVAL);
	}
	if(max_tile_length < 1) {
		XLALPrintError("max_tile_length < 1");
		XLAL_ERROR(XLAL_EINVAL);
	}
	if(fractional_tile_shift <= 0) {
		XLALPrintError("fractional_tile_shift <= 0");
		XLAL_ERROR(XLAL_EINVAL);
	}
	if(fmod(fractional_tile_shift * max_tile_length, 1) != 0) {
		XLALPrintError("fractional_tile_shift * max_tile_length not an integer");
		XLAL_ERROR(XLAL_EINVAL);
	}
	if(max_tile_shift < 1) {
		XLALPrintError("fractional_tile_shift * max_tile_length < 1");
		XLAL_ERROR(XLAL_EINVAL);
	}

	/*
	 * discard first and last 4096 samples
	 *
	 * FIXME.  this should be tied to the sample frequency and
	 * time-frequency plane's channel spacing.  multiplying the time
	 * series by a window has the effect of convolving the Fourier
	 * transform of the data by the F.T. of the window, and we don't
	 * want this to blur the spectrum by an amount larger than 1
	 * channel --- a delta function in the spectrum should remain
	 * confined to a single bin.  a channel width of 2 Hz means the
	 * notch feature created in the time series by the window must be
	 * at least .5 s long to not result in undesired leakage.  at a
	 * sample frequency of 8192 samples / s, it must be at least 4096
	 * samples long (2048 samples at each end of the time series).  to
	 * be safe, we double that to 4096 samples at each end.
	 */

	*window_pad = 4096;

	/*
	 * tiling covers the remainder, rounded down to fit an integer
	 * number of tiles
	 */

	*tiling_length = window_length - 2 * *window_pad;
	*tiling_length = XLALOverlappedSegmentsCommensurate(*tiling_length, max_tile_length, max_tile_shift);
	if(*tiling_length <= 0) {
		XLALPrintError("window_length too small for tiling, must be >= 2 * %d + %d", *window_pad, max_tile_length);
		XLAL_ERROR(XLAL_EINVAL);
	}

	/*
	 * now re-compute window_pad from rounded-off tiling_length
	 */

	*window_pad = (window_length - *tiling_length) / 2;
	if(*tiling_length + 2 * *window_pad != window_length) {
		XLALPrintError("window_length does not permit equal padding before and after tiling");
		XLAL_ERROR(XLAL_EINVAL);
	}

	/*
	 * adjacent tilings overlap so that their largest tiles overlap the
	 * same as within each tiling
	 */

	*window_shift = *tiling_length - (max_tile_length - max_tile_shift);
	if(*window_shift < 1) {
		XLALPrintError("window_shift < 1");
		XLAL_ERROR(XLAL_EINVAL);
	}

	/*
	 * compute the adjusted PSD length if desired
	 */

	if(psd_length) {
		*psd_length = XLALOverlappedSegmentsCommensurate(*psd_length, window_length, *window_shift);
		if(*psd_length < 0)
			XLAL_ERROR(XLAL_EFUNC);

		*psd_shift = *psd_length - (window_length - *window_shift);
		if(*psd_shift < 1) {
			XLALPrintError("psd_shift < 1");
			XLAL_ERROR(XLAL_EINVAL);
		}
	} else if(psd_shift) {
		/* for safety */
		*psd_shift = -1;
		/* can't compute psd_shift without psd_length input */
		XLAL_ERROR(XLAL_EFAULT);
	}

	return 0;
}


/*
 * ============================================================================
 *
 *                   Time-Frequency Plane Create / Destroy
 *
 * ============================================================================
 */


/**
 * Create and initialize a time-frequency plane object.
 */
REAL8TimeFrequencyPlane *XLALCreateTFPlane(
	UINT4 tseries_length,		/**< length of time series from which TF plane will be computed */
	REAL8 tseries_deltaT,		/**< sample rate of time series */
	REAL8 flow,			/**< minimum frequency to search for */
	REAL8 bandwidth,		/**< bandwidth of TF plane */
	REAL8 tiling_fractional_stride,	/**< overlap of adjacent tiles */
	REAL8 max_tile_bandwidth,	/**< largest tile's bandwidth */
	REAL8 max_tile_duration,	/**< largest tile's duration */
	const REAL8FFTPlan *plan	/**< forward plan whose length is tseries_length */
)
{
	REAL8TimeFrequencyPlane *plane;
	gsl_matrix *channel_data;
	REAL8Sequence *channel_buffer;
	REAL8Sequence *unwhitened_channel_buffer;
	REAL8Window *tukey;
	REAL8Sequence *correlation;

	/*
	 * resolution of FT of input time series
	 */

	const double fseries_deltaF = 1.0 / (tseries_length * tseries_deltaT);

	/*
	 * time-frequency plane's channel spacing
	 */

	const double deltaF = 1 / max_tile_duration * tiling_fractional_stride;

	/*
	 * total number of channels
	 */

	const int channels = round(bandwidth / deltaF);

	/*
	 * stride
	 */

	const unsigned inv_fractional_stride = round(1.0 / tiling_fractional_stride);

	/*
	 * tile size limits
	 */

	const unsigned min_length = round((1 / max_tile_bandwidth) / tseries_deltaT);
	const unsigned max_length = round(max_tile_duration / tseries_deltaT);
	const unsigned min_channels = inv_fractional_stride;
	const unsigned max_channels = round(max_tile_bandwidth / deltaF);

	/*
	 * sample on which tiling starts
	 */

	int tiling_start;

	/*
	 * length of tiling
	 */

	int tiling_length;

	/*
	 * window shift
	 */

	int window_shift;

	/*
	 * Compute window_shift, tiling_start, and tiling_length.
	 */

	if(XLALEPGetTimingParameters(tseries_length, max_tile_duration / tseries_deltaT, tiling_fractional_stride, NULL, NULL, &window_shift, &tiling_start, &tiling_length) < 0)
		XLAL_ERROR_NULL(XLAL_EFUNC);

	/*
	 * Make sure that input parameters are reasonable, and that a
	 * complete tiling is possible.
	 *
	 * Note that because all tile durations are integer power of two
	 * multiples of the smallest duration, if the largest duration fits
	 * an integer number of times in the tiling length, then all tile
	 * sizes do so there's no need to test them all.  Likewise for the
	 * bandwidths.
	 *
	 * FIXME:  these tests require an integer number of non-overlapping
	 * tiles to fit, which is stricter than required;  only need an
	 * integer number of overlapping tiles to fit, but then probably
	 * have to test all sizes separately.
	 */

	if((flow < 0) ||
	   (bandwidth <= 0) ||
	   (deltaF <= 0) ||
	   (inv_fractional_stride * tiling_fractional_stride != 1) ||
	   (fmod(max_tile_duration, tseries_deltaT) != 0) ||
	   (fmod(deltaF, fseries_deltaF) != 0) ||
	   (tseries_deltaT <= 0) ||
	   (channels * deltaF != bandwidth) ||
	   (min_length * tseries_deltaT != (1 / max_tile_bandwidth)) ||
	   (min_length % inv_fractional_stride != 0) ||
	   (tiling_length % max_length != 0) ||
	   (channels % max_channels != 0)) {
		XLALPrintError("unable to construct time-frequency tiling from input parameters\n");
		XLAL_ERROR_NULL(XLAL_EINVAL);
	}

	/*
	 * Allocate memory.
	 */

	plane = XLALMalloc(sizeof(*plane));
	channel_data = gsl_matrix_alloc(tseries_length, channels);
	channel_buffer = XLALCreateREAL8Sequence(tseries_length);
	unwhitened_channel_buffer = XLALCreateREAL8Sequence(tseries_length);
	tukey = XLALCreateTukeyREAL8Window(tseries_length, (tseries_length - tiling_length) / (double) tseries_length);
	if(tukey)
		correlation = XLALREAL8WindowTwoPointSpectralCorrelation(tukey, plan);
	else
		/* error path */
		correlation = NULL;
	if(!plane || !channel_data || !channel_buffer || !unwhitened_channel_buffer || !tukey || !correlation) {
		XLALFree(plane);
		if(channel_data)
			gsl_matrix_free(channel_data);
		XLALDestroyREAL8Sequence(channel_buffer);
		XLALDestroyREAL8Sequence(unwhitened_channel_buffer);
		XLALDestroyREAL8Window(tukey);
		XLALDestroyREAL8Sequence(correlation);
		XLAL_ERROR_NULL(XLAL_EFUNC);
	}

	/*
	 * Initialize the structure
	 */

	plane->name[0] = '\0';
	XLALGPSSetREAL8(&plane->epoch, 0.0);
	plane->deltaT = tseries_deltaT;
	plane->fseries_deltaF = fseries_deltaF;
	plane->deltaF = deltaF;
	plane->flow = flow;
	plane->channel_data = channel_data;
	plane->channel_buffer = channel_buffer;
	plane->unwhitened_channel_buffer = unwhitened_channel_buffer;
	plane->tiles.max_length = max_length;
	plane->tiles.min_channels = min_channels;
	plane->tiles.max_channels = max_channels;
	plane->tiles.tiling_start = tiling_start;
	plane->tiles.tiling_end = tiling_start + tiling_length;
	plane->tiles.inv_fractional_stride = inv_fractional_stride;
	plane->tiles.dof_per_pixel = 2 * tseries_deltaT * deltaF;
	plane->window = tukey;
	plane->window_shift = window_shift;
	plane->two_point_spectral_correlation = correlation;

	/*
	 * Success
	 */

	return plane;
}


/**
 * Free a time-frequency plane object.
 */
void XLALDestroyTFPlane(
	REAL8TimeFrequencyPlane *plane
)
{
	if(plane) {
		if(plane->channel_data)
			gsl_matrix_free(plane->channel_data);
		XLALDestroyREAL8Sequence(plane->channel_buffer);
		XLALDestroyREAL8Sequence(plane->unwhitened_channel_buffer);
		XLALDestroyREAL8Window(plane->window);
		XLALDestroyREAL8Sequence(plane->two_point_spectral_correlation);
	}
	XLALFree(plane);
}


/*
 * ============================================================================
 *
 *                         Channel Filter Management
 *
 * ============================================================================
 */


/**
 * Compute the magnitude of the inner product of two arbitrary channel
 * filters.  Note that the sums are done over only the positive frequency
 * components, so this function multiplies by the required factor of 2.
 * The result is the *full* inner product, not the half inner product.  It
 * is safe to pass the same filter as both arguments.  If the PSD is set to
 * NULL then no PSD weighting is applied.  PSD weighting is only used in
 * reconstructing h_rss.
 *
 * The return value is NaN if the input frequency series have incompatible
 * parameters.  Note that the two-point spectral correlation function does
 * not carry enough metadata to determine if it is compatible with the
 * filters or PSD, for example it does not carry a deltaF parameter.  It is
 * left as an excercise for the calling code to ensure the two-point
 * spectral correlation is appropriate.
 */
double XLALExcessPowerFilterInnerProduct(
	const COMPLEX16FrequencySeries *filter1,	/**< frequency-domain filter */
	const COMPLEX16FrequencySeries *filter2,	/**< frequency-domain filter */
	const REAL8Sequence *correlation,		/**< two-point spectral correlation function.  see XLALREAL8WindowTwoPointSpectralCorrelation(). */
	const REAL8FrequencySeries *psd			/**< power spectral density function.  see XLALREAL8AverageSpectrumWelch() and friends. */
)
{
	const int k10 = round(filter1->f0 / filter1->deltaF);
	const int k20 = round(filter2->f0 / filter2->deltaF);
	const COMPLEX16 *f1data = (const COMPLEX16 *) filter1->data->data;
	const COMPLEX16 *f2data = (const COMPLEX16 *) filter2->data->data;
	const double *pdata = psd ? psd->data->data - (int) round(psd->f0 / psd->deltaF) : NULL;
	int k1, k2;
	COMPLEX16 sum = 0;

	/*
	 * check that filters have same frequency resolution, and if a PSD
	 * is provided that it also has the same frequency resolution and
	 * spans the frequencies spanned by the fitlers
	 */

	if(filter1->deltaF != filter2->deltaF || (psd &&
		(psd->deltaF != filter1->deltaF || psd->f0 > min(filter1->f0, filter2->f0) || max(filter1->f0 + filter1->data->length * filter1->deltaF, filter2->f0 + filter2->data->length * filter2->deltaF) > psd->f0 + psd->data->length * psd->deltaF)
	)) {
		XLALPrintError("%s(): filters are incompatible or PSD does not span filters' frequencies", __func__);
		XLAL_ERROR_REAL8(XLAL_EINVAL);
	}

	/*
	 * compute and return inner product
	 */

	for(k1 = 0; k1 < (int) filter1->data->length; k1++) {
		for(k2 = 0; k2 < (int) filter2->data->length; k2++) {
			const unsigned delta_k = abs(k10 + k1 - k20 - k2);
			double sksk = (delta_k & 1 ? -1 : +1) * (delta_k < correlation->length ? correlation->data[delta_k] : 0);

			if(pdata)
				sksk *= sqrt(pdata[k10 + k1] * pdata[k20 + k2]);

			sum += sksk * f1data[k1] * conj(f2data[k2]);
		}
	}

	return 2 * cabs(sum);
}


/**
 * Generate the frequency domain channel filter function.  The filter
 * corresponds to a frequency band [channel_flow, channel_flow +
 * channel_width].  The filter is nominally a Hann window twice the
 * channel's width, centred on the channel's centre frequency.  This makes
 * a sum across channels equivalent to constructing a Tukey window spanning
 * the same frequency band.  This trick is one of the ingredients that
 * allows us to accomplish a multi-resolution tiling using a single
 * frequency channel projection (*).
 *
 * The filter is normalized so that its "magnitude" as defined by the inner
 * product function XLALExcessPowerFilterInnerProduct() is N.  Then the
 * filter is divided by the square root of the PSD frequency series prior
 * to normalilization.  This has the effect of de-emphasizing frequency
 * bins with high noise content, and is called "over whitening".
 *
 * Note:  the number of samples in the window is odd, being one more than
 * the number of frequency bins in twice the channel width.  This gets the
 * Hann windows to super-impose to form a Tukey window.  (you'll have to
 * draw yourself a picture).
 *
 * (*) Really, there's no need for the "effective window" resulting from
 * summing across channels to be something that has a name, any channel
 * filter at all would do, but this way the code's behaviour is more easily
 * understood --- it's easy to say "the channel filter is a Tukey window of
 * variable central width".
 */
COMPLEX16FrequencySeries *XLALCreateExcessPowerFilter(
	REAL8 channel_flow,			/**< Hz */
	REAL8 channel_width,			/**< Hz */
	const REAL8FrequencySeries *psd,	/**< power spectral density function.  see XLALREAL8AverageSpectrumWelch() and friends. */
	const REAL8Sequence *correlation	/**< two-point spectral correlation function.  see XLALREAL8WindowTwoPointSpectralCorrelation(). */
)
{
	char filter_name[100];
	REAL8Window *hann;
	COMPLEX16FrequencySeries *filter;
	unsigned i;
	REAL8 norm;

	/*
	 * create frequency series for filter
	 */

	sprintf(filter_name, "channel %g +/- %g Hz", channel_flow + channel_width / 2, channel_width / 2);
	filter = XLALCreateCOMPLEX16FrequencySeries(filter_name, &psd->epoch, channel_flow - channel_width / 2, psd->deltaF, &lalDimensionlessUnit, 2 * channel_width / psd->deltaF + 1);
	if(!filter)
		XLAL_ERROR_NULL(XLAL_EFUNC);
	if(filter->f0 < 0.0) {
		XLALPrintError("%s(): channel_flow - channel_width / 2 >= 0.0 failed", __func__);
		XLALDestroyCOMPLEX16FrequencySeries(filter);
		XLAL_ERROR_NULL(XLAL_EINVAL);
	}

	/*
	 * build real-valued Hann window and copy into filter
	 */

	hann = XLALCreateHannREAL8Window(filter->data->length);
	if(!hann) {
		XLALDestroyCOMPLEX16FrequencySeries(filter);
		XLALDestroyREAL8Window(hann);
		XLAL_ERROR_NULL(XLAL_EFUNC);
	}
	for(i = 0; i < filter->data->length; i++)
		filter->data->data[i] = hann->data->data[i];
	XLALDestroyREAL8Window(hann);

	/*
	 * divide by square root of PSD to whiten
	 */

	if(!XLALWhitenCOMPLEX16FrequencySeries(filter, psd)) {
		XLALDestroyCOMPLEX16FrequencySeries(filter);
		XLAL_ERROR_NULL(XLAL_EFUNC);
	}

	/*
	 * normalize the filter.  the filter needs to be normalized so that
	 * it's inner product with itself is (width / delta F), the width
	 * of the filter in bins.
	 */

	norm = XLALExcessPowerFilterInnerProduct(filter, filter, correlation, NULL);
	if(XLAL_IS_REAL8_FAIL_NAN(norm)) {
		XLALDestroyCOMPLEX16FrequencySeries(filter);
		XLAL_ERROR_NULL(XLAL_EFUNC);
	}
	norm = sqrt(channel_width / filter->deltaF / norm);
	for(i = 0; i < filter->data->length; i++)
		filter->data->data[i] *= norm;

	/*
	 * success
	 */

	return filter;
}
