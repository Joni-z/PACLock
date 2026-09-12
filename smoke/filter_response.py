"""Check realized first-band response, not just nominal cutoff parameters."""
import torch

@torch.no_grad()
def check_low_band_response(frontend):
    nfft = 16384
    fs = frontend.sinc.sample_rate
    response = torch.fft.rfft(frontend.sinc.filters()[0,0], n=nfft).abs()
    peak, index = response.max(0)
    hz = float(index) * fs / nfft
    center, width = frontend.band_hz()[0].tolist()
    low, high = center-width/2, center+width/2
    if not (low-fs/nfft <= hz <= high+fs/nfft) or float(response[0]) >= float(peak)*(1-1e-6):
        raise AssertionError(f'Lowest filter does not realize its intended band: '
                             f'nominal={low:.4f}..{high:.4f} Hz, peak={hz:.4f} Hz, '
                             f'DC/peak={float(response[0]/peak):.4f}')
    return dict(peak_hz=hz, dc_to_peak=float(response[0]/peak))
