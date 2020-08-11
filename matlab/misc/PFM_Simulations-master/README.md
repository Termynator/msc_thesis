# PFM Simulations

This simulation framework has been used as part of the following two
publications:

 + `Harrison-2015`:
   Harrison et al., 'Large-scale Probabilistic Functional Modes from resting
   state fMRI', NeuroImage, 2015.
   DOI: [10.1016/j.neuroimage.2015.01.013](https://doi.org/10.1016/j.neuroimage.2015.01.013)

 + `Bijsterbosch-2018`:
   Bijsterbosch et al., 'The relationship between spatial configuration and
   functional connectivity of brain regions revisited', bioRxiv, 2018.
   DOI: [10.1101/520502](https://doi.org/10.1101/520502)

The main script is [RunTests.m](RunTests.m).

```shell
module load MATLAB/current  # Requires MATLAB >= R2016b
matlab -nodesktop -nosplash -r RunTests
```

### Requirements

 + MATLAB >= R2016b
 + `PROFUMO` and `melodic` must be installed and available on `$PATH`.
 + The above will not work on Windows, though it should be possible to generate
   data (after some adjustments to filesystem paths).

----------

Copyright 2020 University of Oxford, Oxford, UK.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
