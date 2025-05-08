import numpy as np
import numpy.random as ra
import scipy.fftpack as sf
from astropy.io import fits
from astropy.wcs import WCS

R_EARTH = 6.371e6  # Earth radius in meters

class TECScreens:
    """
    Generate 2D TEC screens with frozen-flow drift and AR(1) temporal coherence.
    """
    def __init__(self, nx, ny, dx, dy, dt, p=3.0, L0=200e3, l0=1e3,
                 var=4.0,  # TEC variance in TECU^2
                 v=(300.0, 0.0),  # drift velocity (m/s)
                 tau=60.0,  # coherence time (s)
                 seed=None):
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.dt = dt
        self.alpha = np.exp(-dt / tau)
        self.var = var
        self.vx, self.vy = v
        # frequency grids scaled by layer grid spacing
        fx = np.fft.fftfreq(nx, dx)
        fy = np.fft.fftfreq(ny, dy)
        self.kx, self.ky = np.meshgrid(fx, fy, indexing='ij')
        self.k2 = self.kx**2 + self.ky**2
        # build Von Kármán filter
        self.filter = self._make_filter(p, L0, l0)
        self.prev_ft = None
        if seed is not None:
            ra.seed(seed)

    def _make_filter(self, p, L0, l0):
        k2 = self.k2.copy()
        k2[0,0] = 1e-12
        k0 = 1.0 / L0
        ki = 1.0 / l0
        psd = (k2 + k0**2)**(-p/2) * np.exp(-k2 / ki**2)
        return np.sqrt(psd)

    def step(self):
        noise = ra.normal(size=(self.nx, self.ny))
        noise_ft = sf.fft2(noise) * self.filter
        scale = np.sqrt(1 - self.alpha**2)
        if self.prev_ft is None:
            new_ft = noise_ft * np.sqrt(self.var / np.mean(np.abs(noise_ft)**2))
        else:
            phase_ramp = np.exp(-2j * np.pi * (self.kx * self.vx + self.ky * self.vy) * self.dt)
            drifted = self.prev_ft * phase_ramp
            new_ft = (self.alpha * drifted +
                      scale * noise_ft * np.sqrt(self.var / np.mean(np.abs(noise_ft)**2)))
        self.prev_ft = new_ft
        return sf.ifft2(new_ft).real

    def compute_efolding(self, screens):
        ts = screens[:, self.nx//2, self.ny//2]
        ac = []
        for lag in range(len(ts)//2):
            ac.append(np.corrcoef(ts[:-lag or None], ts[lag:])[0,1])
        ac = np.array(ac)
        lags = np.arange(len(ac)) * self.dt
        thresh = np.exp(-1)
        idx = np.where(ac <= thresh)[0]
        return (lags[idx[0]] if idx.size else None), lags, ac

class MultiLayerTECScreens:
    """
    Combine multiple TECScreens layers with height-dependent sampling.
    """
    def __init__(self, nx, ny, dx_ground, dy_ground, dt, layer_params,
                 p=3.0, L0=200e3, l0=1e3, tau=60.0, seed=None):
        """
        layer_params: list of tuples (var, speed, direction_deg, height_m)
        dx_ground, dy_ground: ground-projected grid spacing (m)
        """
        self.layers = []
        for i, (layer_var, speed, direction, height) in enumerate(layer_params):
            # compute layer-specific grid spacing via Earth curvature
            scale = (R_EARTH + height) / R_EARTH
            dx_i = dx_ground * scale
            dy_i = dy_ground * scale
            # drift velocity vector
            rad = np.deg2rad(direction)
            v = (speed * np.cos(rad), speed * np.sin(rad))
            # instantiate TECScreens for this layer
            scr = TECScreens(nx, ny, dx_i, dy_i, dt, p, L0, l0,
                              var=layer_var, v=v, tau=tau,
                              seed=(seed+i if seed is not None else None))
            self.layers.append((scr, height))
        self.nx, self.ny, self.dt = nx, ny, dt

    def run(self, nsteps, verbose=False):
        data = np.zeros((nsteps, self.nx, self.ny))
        for t in range(nsteps):
            if verbose:
                print(f"Step {t+1}/{nsteps}")
            frame = np.zeros((self.nx, self.ny))
            for scr, _ in self.layers:
                frame += scr.step()
            data[t] = frame
        return data

    def write_fits(self, filename, screens, frequency=1e8):
        nt = screens.shape[0]
        data = screens[np.newaxis, ...]
        w = WCS(naxis=4)
        w.wcs.cdelt = [self.layers[0][0].dx, self.layers[0][0].dy, self.dt, 1.0]
        w.wcs.crpix = [self.nx//2+1, self.ny//2+1, nt//2+1, 1.0]
        w.wcs.ctype = ["LON", "LAT", "TIME", "FREQ"]
        w.wcs.crval = [0.0, 0.0, 0.0, frequency]
        header = w.to_header()
        # record layer heights in header
        for idx, (_, h) in enumerate(self.layers, start=1):
            header[f'HDR{idx:02d}'] = (h, f'Layer {idx} height (m)')
        fits.writeto(filename, data, header=header, overwrite=True)

if __name__ == '__main__':
    nx, ny = 256, 256
    dx_ground = dy_ground = 100.0  # m
    dt = 60.0  # s
    # define F- and E-region layers
    layer_params = [
        # var_TECU2, speed(m/s), direction(deg), height(m)
        (4.0, 300.0, 45.0, 350e3),  # F-region
        (1.0, 100.0, -30.0, 110e3)   # E-region
    ]
    ml = MultiLayerTECScreens(nx, ny, dx_ground, dy_ground, dt, layer_params,
                              p=3.0, L0=200e3, l0=1e3, tau=60.0, seed=42)
    screens = ml.run(240, verbose=True)
    tau_eff, lags, ac = ml.layers[0][0].compute_efolding(screens)
    print(f"Effective e-folding time: {tau_eff} s")
    ml.write_fits('tec_ionoscreen.fits', screens)
