# LS2D

<!-- ----------------------- -->
<!--     PROJECT SHIELDS     -->
<!-- ----------------------- -->
[![Build][build-shield]][none-url]
[![Coverage][coverage-shield]][none-url]
[![Documentation][documentation-shield]][none-url]
[![Website][website-shield]][none-url]
[![Python][python-shield]][none-url]
[![Issues][issues-shield]][none-url]
[![MIT License][license-shield]][none-url]
[![Contributors][contributors-shield]][none-url]

<!--
[![Forks][forks-shield]][none-url]
[![Stargazers][stars-shield]][none-url]
[![MIT License][license-shield]][none-url]
-->

Community | Documentation | Resources | Contributors | Release Notes

Lightweight tool that allows to create embeddings from complex data into 2D
and facilitates the visualisation of the produced latent space. In addition,
it allows to query data and visualise demographics.

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

Pending....

<!-- ----------------------- -->
<!--     GETTING STARTED     -->
<!-- ----------------------- -->
## Getting Started

Pending....


### Docker

$ docker-compose build

$ docker-compose up

### Heroku

https://devcenter.heroku.com/categories/deploying-with-docker

#### Building and pushing image(s)

##### Build an image and push

To build an image and push it to Container Registry, make sure that your directory 
contains a Dockerfile. Note that this will not inspect the docker-compose.yml nor 
the heroku.yml files and therefore the app should be run in the Docker file:
 
Thus, in the Dockerfile include:
 
```console
CMD ["python", "server.py"]
```
 
Run the container to create the image:

```console
heroku container:push <process-type> 
```

For this example

```console
heroku container:push web 
```

##### Pushing an existing image

To push an image to Heroku, such as one pulled from Docker Hub o create with your 
your docker-compose instructions, tag it and push it according to this naming 
template.

```console
docker tag <image> registry.heroku.com/<app>/<process-type>
docker push registry.heroku.com/<app>/<process-type>
```

For this example

```console
docker tag main_web_1 registry.heroku.com/ls2d-demo/web
docker push registry.heroku.com/ls2d-demo/web
```

By specifying the process type in the tag, you can release the image using the CLI. 

```console
heroku container:release web
```

## Other


<!-- ----------------------- -->
<!--     USAGE EXAMPLES      -->
<!-- ----------------------- -->
### Usage

See the [documentation]() for a list of examples.

```py
// Example!

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

#### Contributors (optional)
#### Support (optional)
#### FAQ (optional)