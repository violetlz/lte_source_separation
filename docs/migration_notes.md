# Migration notes

The legacy sibling repository was inspected at commit `a39a2fc2312fc6b5e4cdf0d1962a92f711ce627f`. Its LTE Oracle loader confirmed the historical `[14,2048]` CP-stripped grid and explicit mapping audits, but that path used fixed Oracle per-RE distributions and a different experiment boundary.

This project independently implements a continuous time-domain 5/20 MHz source distribution, valid asynchronous crop, fixed ensemble normalization, typed receiver/truth separation, and independent in-band calibration. It has no `sys.path` modification, relative import, checkpoint load, dataset read, or runtime dependency on the old repository. No old result is presented as a new-platform result.
