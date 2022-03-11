# Latent Space 2D (LS2D)


<img src="docs/logos/logo-ls2d-v1.png" align="right" width="250">

<!-- ----------------------- -->
<!--     PROJECT SHIELDS     -->
<!-- ----------------------- -->
<!--
[![Build][build-shield]][none-url]
[![Coverage][coverage-shield]][none-url]
[![Documentation][documentation-shield]][none-url]
[![Website][website-shield]][none-url]
-->
[![Python][python-shield]][none-url]
[![Issues][issues-shield]][none-url]
[![MIT License][license-shield]][none-url]

<!--
[![Contributors][contributors-shield]][none-url]
-->

<!--
[![Forks][forks-shield]][none-url]
[![Stargazers][stars-shield]][none-url]
[![MIT License][license-shield]][none-url]
-->

Community | Documentation | Resources | Contributors | Release Notes

LS2D is a lightweight tool to create embeddings from complex data into two
dimensions. In addition, it has a web app that (i) facilitates performance 
comparison among the pipelines created, (ii) enables visualisation of the
observations and the distribution of the features/outcomes and (iii) allows
to query patients based on distance and displays a demographic table. 


<!-- Demonstration video -->
https://user-images.githubusercontent.com/1579887/157911761-e74bdc2e-7fe6-4b37-8b34-e260b733e410.mp4

<!-- > Subtitle or Short Description Goes Here -->

<!-- > ideally one sentence -->

<!-- > include terms/tags that can be searched -->


<!-- PROJECT LOGO -->
<!--
<br />
<p align="center">
  <a href="">
    <img src="" alt="Logo" width="150" height="80">
  </a>
</p>
-->


<!-- ----------------------- -->
<!--    TABLE OF CONTENTS    -->
<!-- ----------------------- -->
## Table of Contents

* [About the project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Usage](#usage)
  * [Tests](#tests)
* [Roadmap](#roadmap)
* [License](#license)
* [Contact](#contact)
* [Utils](#utils)

<!--* [Contributing](#contributing)-->
<!--* [Versioning](#versioning)-->
<!--* [Sponsors](#sponsors)-->
<!--* [Authors](#authors)-->
<!--* [Acknowledgements](#acknowledgements)-->

<!-- ----------------------- -->
<!--    ABOUT THE PROJECT    -->
<!-- ----------------------- -->
## About the project

This work aims to facilitate the development and validation of unsupervised techniques to
reduce data complexity to a 2D space so that the information can be relayed to the end user 
through accessible graphical representations. In addition to traditional technices such as 
PCA, autoencoders, a type of neural network, have been used in the included examples.

Live demo (Heroku)**: <a href="http://ls2d-demo.herokuapp.com/" target="_blank"> Link</a>

When using any of this project's source code, please cite:

```console
@article{xxx,
  title = {xxx},
  author = {xxx},
  doi = {xxx},
  journal = {xxx},
  year = {xxx}
}
```

** Heroku puts processes to sleep after certain period of inactivity. Thus, it might take some 
5-7 seconds to load.

<!-- ----------------------- -->
<!--     GETTING STARTED     -->
<!-- ----------------------- -->
## Getting Started

## Training models

First, create a yaml configuration file (see [settings.iris.yaml](ls2d-settings-file)) to define
the data and output location, the features for training, the targets to compute performance metrics,
and the estimators and/or hyperparameters to consider during the grid search. 

Once the configuration is completed, run the search script

```console
$ python search.py --yaml_file
```
  
A new workbench will be created in the output folder containing (i) the generated models 
saved as pickle (.p) files, (ii) the metrics obtained aggregated in the 'results.csv' 
file and the (iii) settings configuration.

To browse through all these results run the ls2d flask app:

```console
$ python server.py
```


### Local

### Using Docker

First build and run the docker container:

```console
$ docker-compose build # Build
$ docker-compose up    # Run
```

Access the app at [https://localhost:5000](https://localhost:5000)

### Deploy to Heroku

#### Building and pushing image(s)

##### Build an image and push

To build an image and push it to Container Registry, make sure that your directory 
contains a Dockerfile. Note that this will not inspect the docker-compose.yml nor 
the heroku.yml files and therefore the app should be run in the Docker file:
 
Thus, in the Dockerfile include:
 
```console
$ CMD ["python", "server.py"]
```
 
Run the container to create the image:

```console
$ heroku container:push <process-type> 
```

For this example

```console
$ heroku container:push web 
```

##### Pushing an existing image

To push an image to Heroku, such as one pulled from Docker Hub o create with your 
your docker-compose instructions, tag it and push it according to this naming 
template.

```console
$ docker tag <image> registry.heroku.com/<app>/<process-type>
$ docker push registry.heroku.com/<app>/<process-type>
```

For this example

```console
$ docker tag main_web_1 registry.heroku.com/ls2d-demo/web
$ docker push registry.heroku.com/ls2d-demo/web
```

By specifying the process type in the tag, you can release the image using the CLI. 

```console
$ heroku container:release web
```

For more information, check the heroku official page 
[here](https://devcenter.heroku.com/categories/deploying-with-docker).


## Other


<!-- ----------------------- -->
<!--     USAGE EXAMPLES      -->
<!-- ----------------------- -->
### Usage

See the [documentation]() for a list of examples.


#### Training models

First, create a yaml configuration file (see [settings.iris.yaml](ls2d-settings-file)).

Run the search script

```console
$ python search.py
```
  
A new workbench will be created in the output folder containing (i) the generated models 
saved as pickle (.p) files, (ii) the metrics obtained aggregated in the 'results.csv' 
file and the (iii) settings configuration.

To browse through all these results run the ls2d flask app:

```console
$ python server.py
```

<!-- ----------------------- -->
<!--          TESTS          -->
<!-- ----------------------- -->
### Tests

This section is empty.


<!-- ----------------------- -->
<!--        ROADMAP          -->
<!-- ----------------------- -->
## Roadmap

See the [open issues]() for a list of proposed features (and known issues).


<!-- ----------------------- -->
<!--         LICENSE         -->
<!-- ----------------------- -->
## License

Distributed under the GNU v3.0 License. See `LICENSE` for more information.

<!-- ----------------------- -->
<!--         CONTACT         -->
<!-- ----------------------- -->
## Contact

Bernard Hernandez - 
   - add twitter
   - add email
   - add linkedin
   - add anything

[Project Link](https://github.com/bahp/ls2d)


<!-- ----------------------- -->
<!--     ACKNOWLEDGEMENTS    -->
<!-- ----------------------- -->
## Acknowledgements

<!-- ----------------------- -->
<!-- MARKDOWN LINKS & IMAGES -->
<!-- ----------------------- -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/badge/contributors-1-yellow.svg
[forks-shield]: https://img.shields.io/badge/forks-0-blue.svg
[stars-shield]: https://img.shields.io/badge/stars-0-blue.svg
[issues-shield]: https://img.shields.io/badge/issues-3_open-yellow.svg
[license-shield]: https://img.shields.io/badge/license-GNUv0.3-orange.svg
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[product-screenshot]: images/screenshot.png

[build-shield]: https://img.shields.io/badge/build-none-yellow.svg
[coverage-shield]: https://img.shields.io/badge/coverage-none-yellow.svg
[documentation-shield]: https://img.shields.io/badge/docs-none-yellow.svg
[website-shield]: https://img.shields.io/badge/website-none-yellow.svg
[python-shield]: https://img.shields.io/badge/python-3.6|3.7|3.8-blue.svg
[pypi-package]: https://img.shields.io/badge/pypi_package-0.0.1-yellow.svg

[dependency-shield]: http://img.shields.io/gemnasium/badges/badgerbadgerbadger.svg?style=flat-square
[coverage-shield]: http://img.shields.io/coveralls/badges/badgerbadgerbadger.svg?style=flat-square
[codeclimate-shield]: http://img.shields.io/codeclimate/github/badges/badgerbadgerbadger.svg?style=flat-square
[githubissues-shield]: http://githubbadges.herokuapp.com/badges/badgerbadgerbadger/issues.svg?style=flat-square
[pullrequests-shield]: http://githubbadges.herokuapp.com/badges/badgerbadgerbadger/pulls.svg?style=flat-square
[gemversion-shield]: http://img.shields.io/gem/v/badgerbadgerbadger.svg?style=flat-square
[license-shield]: http://img.shields.io/:license-mit-blue.svg?style=flat-square
[badges-shield]: http://img.shields.io/:badges-9/9-ff6799.svg?

[none-url]: https://www.imperial.ac.uk/bio-inspired-technology/research/infection-technology/epic-impoc/

[ls2d-settings-file]: https://github.com/bahp/ls2d/blob/main/datasets/iris/settings.iris.yaml

#### Contributors (optional)
#### Support (optional)
#### FAQ (optional)
