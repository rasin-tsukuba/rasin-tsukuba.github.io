---
layout: post
title: Color Theory and Applications Notes 1
subtitle: Chapter 1 The Nature of Color
date: 2020-06-08
author: Rasin
header-img: img/color-theory-1.jpg
catalog: true
tags:
  - Color Theory
  - Physics
  - Human Vision System
---
**This is a series of study notes of the book *Color Vision and Colorimetry Theory and Applications* by *Daniel Malacara*.**

# The Nature of Color

## Introduction

Electromagnetic waves can have many different wavelengths and frequencies in a range known as the electromagnetic spectrum, as illustrated in Fig. 1.1.

![Electromagnetic spectrum](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200608212840.png)

The longest wavelengths produce the perception of red, while the shortest ones produce the perception of violet.

The spectrum in the visible, ultraviolet (UV), and infrared (IR) regions is classified in Table 1.1.

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200608212933.png)

However, the scientific beginning of color studies goes back only to Newton when he performed his classic experiment with a prism, as shown in Fig. 1.2.

![Newton’s prism experiment](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200608213011.png)

The sensation of color is produced by the physical stimulation of the light detectors — known as cones — in the human retina.

Each color has a different wavelength.

The wavelength values corresponding to each color in the spectrum produced by a prism are quite nonlinear. The diffraction grating produces a more linear spectrum.

A spectrally pure or monochromatic color can be produced by a single wavelength. However, the same color can be produced with a combination of two light beams. In conclusion, when we refer to a spectrally pure light beam it does not mean that it is formed by a single-wavelength beam. Instead, it means that it has the same color as the single-wavelength light beam matching its color.

The two or more components used to produce a color cannot be identified by the eye, only with an instrument called a spectroscope. For this reason we say that the eye is a synthesizer device. In contrast, when the ear listens to an orchestra, the individual instruments producing the sound can be identified. Thus, we say that the ear is an analyzer.

Colors arise due to the interaction of light with material bodies. 

The information contained in the light coming from the world around us is not only in the intensity, but also in its color. The human eye can distinguish about 200 different levels of gray, but the number of different possible combinations that the human eye can discriminate greatly increases with color vision, thereby expanding the amount of information that can be extracted from a scene.

Not all colors in nature are spectrally pure, since they can be mixed with white. Colors obtained by mixing a spectrally pure color with white are said to have the same **hue** but different **saturation**. The degree of saturation is called the **chroma**.

Combinations of spectrally pure colors and white cannot produce all possible colors in nature. Therefore, any color has to be specified by three parameters, i.e., **hue, saturation (or chroma), and luminance**, or any other three equivalent parameters, as will be described later in more detail.

## Newton’s Color Experiment

The first experiment in color, performed with a prism by Newton in 1671, demonstrated color dispersion. He used a triangular prism, as illustrated in Fig. 1.2, in a position so that a narrow beam of sunlight entering into the room passed through the prism. When this beam was projected onto a screen, a band of light with different colors appeared, forming what he called a spectrum.

This experiment immediately suggested the idea that white light is formed by the superposition of all colors. To prove this idea, Newton used another prism to recombine all colors from the spectrum into a white beam of light. Newton was very careful to state that the spectrum colors are not the only ones in nature.

The first color diagram was devised by Newton by drawing a circle listing all of the colors of the spectrum. Both spectrally pure colors and purple colors were drawn around the circle with white at the center, as shown in Fig. 1.5.

![Newton’s circle](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200609093054.png)

Colors formed by a mixture of the pure spectral colors and white are between the center and the circumference. Complementary colors are on opposite points with respect to the center. If this circle is made on a piece of cardboard and placed on the axis of a rotating top, all colors mix to produce the impression of a colorless gray.

## Theories and Experiments in Color Vision

Mariotte (1717) said that three colors are sufficient to produce any color when using the proper combination of them. This concept was formally reintroduced by Palmer (1777) in a manuscript that was discovered in 1956, adding the concept of three different color receptors in the retina. This so-called trichromatic theory of color was found satisfactory, but it did not explain many details.

Hering (1964) noticed that yellow and blue appear to be opposite colors. Hering assumed that the brain has a detector for yellow and blue light, followed by a classifier to determine the relative luminance of these two colors.

In the 1950s, Land made an impressive modification to the three-projector experiment by Maxwell. Land was able to project a full-color scene with only two colors, such as red and green or even red and white, as shown in Fig. 1.7. According to the trichromatic theory, a picture with only different shades of pink should be observed if only a red and a white projector are used. Land’s important conclusion is that the results from the classical trichromatic theory are valid only when the observed color samples are surrounded by a dark environment, and that important deviations from the classical results appear when many colored objects are simultaneously observed. Land’s theory and experiment can explain our capacity to observe undistorted colors under a large variety of illumination conditions.

To complicate the color vision models even further, it is easy to show that flickering white light, produced by the alternation of bright and dark fields, can produce color sensations that depend on the flickering frequency. By means of flickering at frequencies smaller than 60 Hz, the so called subjective or Fetchner colors can be produced. These can be produced with rotating disks with many different patterns. Probably the best known is Benham’s disk as shown in Fig. 1.8. Colored rings are observed when the disk is rotated at a speed of 5–10 revolutions per second.

![Benham’s disk with the light-intensity distribution as a function of time](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200609101243.png)

Since the pattern is not rotationally symmetrical, different colors are observed not only for different speeds, but also for different directions of rotation.

Another interesting effect related to flickering is the McCollough effect, as illustrated in Fig. 1.9. The first two patterns, with vertical and horizontal lines, respectively, should be observed for 10 min in an alternating manner, for ~10 sec each, and then the third pattern is observed. Pale green will be seen where the vertical lines are and pale red where the horizontal lines are. In conclusion, the perception of color is not a purely physical mechanism but also physiological and psychological.

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200609101814.png)

## Some Radiometric and Photometric Units

In this section, before going to more interesting matters, we will begin the tedious work of defining some of the most important quantities used in colorimetry studies.

### Radiometric units

The most basic definition is that of **radiant flux** \\(\phi_{R}\\), which is the radiant energy per unit time (power) transported by a light beam, measured in watts (\\(J/s\\)). 

The **irradiance** or **radiant incidence** \\(E\\) is the *area density of radiant flux received* by an illuminated body, integrated for all wavelengths and all directions. It is measured in *watts per square meter*, and defined by:

$$
E = \frac{radiant flux}{area} = \frac{d\phi_R}{dA}
$$

The *spectral irradiance* \\(E(\lambda)\\), defined as the irradiance per unit wavelength interval, at the wavelength \\(\lambda\\), is related to irradiance \\(E\\) by the relation:

$$
E = \int^\infty_0 E(\lambda)d\lambda
$$

Similar to irradiance, or radiant incidence, the *radiant exitance* \\(M\\) of an extended light source, defined as the area density of the total radiant flux emitted and integrated for all wavelengths and all directions, is

$$
M = \frac{emitted \ radiant\  flux}{area} = \frac{d\phi_R}{dA}
$$

The *radiant intensity* \\(I\\) of a point light source size, defined as the total radiant flux, integrated for the whole area of the source and all wavelengths, emitted per unit solid angle in a given direction, is

$$
I = \frac{d\phi_R}{d\Omega}
$$

The *spectral radiant intensity* \\(I(\lambda)\\) defined as the radiant intensity per unit wavelength interval, at the wavelength \\(\lambda\\), is related to the radiant intensity \\(I\\) by

$$
I = \int^\infty_0 I(\lambda)d\lambda
$$

Thus, the spectral radiant intensity is, in general, a function of the direction as well as of the wavelength.

The *radiance* \\(L\\) of an extended light source, defined as the radiant flux per unit solid angle, per unit of projected area, in a given direction, integrated for all wavelengths, is

$$
L = \frac{radiant\ flux\ per\ steradian}{projected area} = \frac{d^2\phi_R}{dA_Pd\Omega} = \frac{1}{cos \theta} \frac{d^2\phi R}{dAd\Omega}
$$

where the projected area \\(A_P\\) is equal to the area \\(A\\) of the source multiplied by the cosine of the angle \\(\theta\\) between the line of sight and the normal to the source. The radiance is a function of the observing direction \\(\theta\\), but for perfectly illuminated diffusing surfaces, called Lambert surfaces, this is a constant.

The *spectral radiance* \\(L(\lambda)\\), defined as the radiance per unit wavelength interval, at the wavelength \\(\lambda\\), is related to the radiance \\(L\\) by

$$
L = \int^\infty_0 L(\lambda)d\lambda
$$

Table 1.2 summarizes these units.

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200609112417.png)

[Chinese Version of Radiance Unit](https://zh.wikipedia.org/wiki/%E8%BE%90%E5%B0%84%E5%BA%A6%E9%87%8F%E5%AD%A6)

### Photometric Units

When light enters the eye, a luminous stimulus is produced.

The *luminous flux* \\(\phi_v\\) can be considered the basic photometric unit, equivalent to the radiant flux, but evaluated according to the magnitude of the luminous stimulus it produces,measured in lumens (lm).

This luminous stimulus is strongly dependent on wavelength, as will be noted later in this section.

The *illuminance* \\(E_v\\) is the area density of luminous flux received by a illuminated body, integrated for all wavelengths and all directions. Its unit is the *lux*, in lumens per square meter, and is defined by 

$$
E_v = \frac{luminous \ flux}{area} =\frac{d\phi_v}{dA}
$$

The *spectral illuminance* \\(E_v(\lambda)\\) is defined as the illuminance per unit wavelength interval, at the wavelength \\(\lambda\\), and is related to the illuminance \\(E_v\\) by the relation

$$
E_v=\int_0^\infty E_v(\lambda)d\lambda
$$

The *luminous exitance* \\(M_v\\) of an extended light source, defined as the area density of the total luminous flux emitted, integrated for all wavelengths and all directions, is

$$
M_v = \frac{emitted\ luminous\ flux}{area} = \frac{d\phi_v}{dA}
$$

which is measured in lumens per square meter (not luxes, which are used only for illuminance). This quantity is rarely used in colorimetry.

The *luminous intensity* \\(I_v\\) of a point light source, defined as the total luminous flux, integrated for the whole area of the source and all wavelengths, emitted per unit solid angle in a given direction, is

$$
I_v=\frac{d\phi_v}{d_\Omega}
$$

From the beginning of the nineteenth century, the unit of luminous intensity was the candle. In 1979 it was redefined and confirmed with the Latin name *candela* (cd). A candela is defined as the luminous intensity in a given direction of a source that emits monochromatic light of frequency \\(540 \times 1012\\) Hz (wavelength equal to 555 nm) and has a radiant intensity in that direction of \\(\frac{1}{683}\ W/sr\\). This average is now known as the standard observer.

With this definition the lumen becomes a secondary unit defined in terms of the *candela*. So, one lumen is equal to one candela emitted per unit solid angle (steradian, abbreviated as sr).

The *spectral luminous intensity* \\(I_v(\lambda)\\) of a point light source is defined as the luminous intensity per unit wavelength interval, at the wavelength \\(\lambda\\), and is related to the luminous intensity \\(I_v\\) by

$$
I_v=\int_0^\infty I_v(\lambda)d\lambda
$$

The *luminance* \\(L_v\\) of a luminous object is defined as the total integrated luminous flux for all wavelengths, emitted per unit solid angle, per unit of projected area in a given direction, of the luminous surface, as follows:

$$
L_v = \frac{luminous\ flux\ per\ steradian}{projected\ area} = \frac{d^2\phi_v}{dA_pd\Omega} = \frac{1}{cos \theta}\frac{d^2\phi_v}{dAd\Omega}
$$

where the projected area \\(A_p\\) is equal to the actual area \\(A\\) of the source, multiplied by the cosine of the angle\\(\theta\\) between the line of sight and the normal to the source. As with the radiance, the luminance is also a function of the direction of observation.

The *spectral luminance* \\(L_v(\lambda)\\), defined as the luminance per unit wavelength interval, at the wavelength \\(\lambda\\), is related to the luminance \\(L_v\\) by

$$
L_v=\int_0^\infty L_v(\lambda)d\lambda
$$

Given a certain spectral distribution, the **luminance** is directly proportional to the radiance, and there is a linear relation between them. however, the eye response is not linear to either of these two quantities.

Another quantity, called **lightness**, which is nonlinear to the luminance but linear to the eye response.

The **brightness** is not a physical unit but a more subjective term related to the perception elicited by the luminance of a luminous object.

An example is illustrated in Fig. 1.10, where a strip with constant luminance is surrounded by a field with a gradient in luminance. The inner strip has a gradient brightness but a constant luminance.

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200610212012.png)

Table 1.3 summarizes these units.

![](https://raw.githubusercontent.com/rasin-tsukuba/blog-images/master/img/20200610212553.png)