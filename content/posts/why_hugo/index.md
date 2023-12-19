+++
title = 'Why Hugo instead of Jekyll?'
date = 2023-12-19T10:16:13Z
tags = ['hugo', 'development']
+++

I built this website as a platform for giving updates on my projects and sharing my thoughts. As someone who builds software and interacts with GitHub on a daily basis, the natural choice for hosting a website was GitHub pages. It’s completely free and you don’t even have to create another account. The only limitation is that it can only host static sites, but that’s not really an issue for a simple text-heavy websites such as blogs.

## What are static site generators?

Even after making this decision, there are many different potential tools for generating the site. Unless you’re a web developer, the best option for most people is to use a static-site generator (SSG). These tools allow you to write your content in markdown, and then automatically generate the website without having to touch HTML. I found that there are broadly two types of SSGs:

| "Simple" generators                                   | "Advanced" generators               |
| ----------------------------------------------------- | ----------------------------------- |
| [Jekyll](https://jekyllrb.com/) (written in Ruby)     | [Gatsby](https://www.gatsbyjs.com/) |
| [Hugo](https://gohugo.io/) (written in Go)            | [Astro](https://astro.build/)       |
| [11ty](https://www.11ty.dev/) (written in JavaScript) |                                     |

The "advanced" options also support some dynamic content and are targeted for “proper” websites. I therefore instantly discounted these options, as they would introduce unnecessary complexity for a blog in my opinion.

Of the "simple" options, Jekyll is anecdotally the most popular, likely because this has been the default option from the start of GitHub pages. Hugo is a newer option and has recently overtaken Jekyll in terms of stars. 11ty is newer still but much less popular.

[![Star History Chart](https://api.star-history.com/svg?repos=jekyll/jekyll,gohugoio/hugo,11ty/eleventy&type=Date)](https://star-history.com/#jekyll/jekyll&gohugoio/hugo&11ty/eleventy&Date)

I decided to immediately discount 11ty since it appears to be more targeted towards those with more frontend experience and has a smaller community, so my remaining choices were Jekyll and Hugo. It seems that the consensus is that Jekyll is “simpler” and appears to still be considered to be the “default” option. However, things have changed a lot since GitHub pages was first introduced and my experience with Hugo has been positive (so far). Therefore, I thought it would be useful to share my reasoning for choosing Hugo in case it helps someone else.

# So why did I choose Hugo?

## Hugo ships as a single executable

Jekyll is shipped as a Ruby package, which means you first need to install Ruby on your machine. I remember how much of a pain it was to set up a Python environment for the first time [^1], so the idea of having to install Ruby just seemed like an unnecessary headache.

[^1]: See obligatory reference to this [XKCD comic](https://xkcd.com/1987/)

Hugo on the other hand ships as a single executable so you can do `brew install hugo`. It’s just so much easier, and was one of the biggest reasons why Hugo initially appealed to me.

## Hugo’s own documentation is excellent

Since Jekyll is over 10 years old, and you can find loads of examples online to help resolve issues if you get stuck. With Hugo, when I do search for issues that arise, I often get results for Jekyll mixed in, so this is certainly one area where Hugo is weaker.

Fortunately, Hugo’s website itself is very good. The API [documentation](https://gohugo.io/documentation/) is quite extensive, and the [forums](https://discourse.gohugo.io/) are active. I will concede though that the documentation is quite technical and assumes a lot of prior knowledge, so there are a lot of concepts to get your head around initially (although I don’t know if this is any worse than Jekyll).

While think it is slightly harder to find help at the moment in Hugo, I think this issue is generally overstated. Given the popularity of Hugo, this is something which should continue to improve over time.

## Hugo shortcodes are deceptively powerful

I read in multiple places that the leaning curve for custom templates is harder with Hugo than Jekyll. For simple templating, Jekyll’s [includes](https://jekyllrb.com/docs/includes/) are arguably simpler and easier to read using [Liquid tags](https://shopify.github.io/liquid/).

Hugo’s [shortcodes](https://gohugo.io/content-management/shortcodes/) are more verbose than includes, but they support more complex logic. I’m sure you can achieve _even more_ with a Jekyll [plug-in](https://jekyllrb.com/docs/plugins/) if you are proficient in Ruby, but I was able to create a few custom components with shortcodes pretty quickly without any prior experience with Go.

To me it seems that shortcodes strike a sensible middle ground which is _powerful enough_ for _most_ users.

## Deploying Hugo websites with Github Actions is easy

If you use Jekyll, you can websites to GitHub pages using the Ruby environment provide by GitHub Pages [Ruby gem](https://pages.github.com/versions/). This means that you just push your changes to a specific branch and the website automatically gets built and deployed without any additional configuration.

There is an alternative method using [GitHub actions](https://jekyllrb.com/docs/continuous-integration/github-actions/), where you configure a workflow to run when certain criteria are met (normally a commit to the `main` branch). The action then builds your website and deploys it [^2].

[^2]: It seems that GitHub actions will likely become the new default for Jekyll too, given that GitHub [still hasn’t updated pages to use Jekyll 4](https://github.com/github/pages-gem/issues/651).

With Hugo on the other hand, you _have_ to use GitHub actions. If you’ve never used CI before, this might seem daunting, but Hugo has [clear instructions](https://gohugo.io/hosting-and-deployment/hosting-on-github/) on how to configure everything. They also provide a template workflow, which you can simply copy over to your repo. It took me about 10 minutes to first deploy the first version of this website.

Personally I prefer using GitHub actions because:

1. It is more explicit how the site is deployed, and you have full control over when and how it is built and deployed
2. It avoids having to learn how the Ruby environment provided by GitHub pages works
3. It allows me to also configure build checks on every commit (even those not on `main`)[^3]

[^3]: You can view my modified workflow [here](https://github.com/alxhslm/alxhslm.github.io/blob/main/.github/workflows/hugo.yaml).

## Hugo is faster

This is unlikely to really matter for most people, but since Hugo is written in Go which is a compiled language, it builds very quickly and [faster than Jekyll](https://michaelnordmeyer.com/benchmarking-hugo-vs-jekyll-vs-github-pages-in-2023#:~:text=Hugo%20is%20faster%20than%20Jekyll,50%20%25%20for%20higher%20post%20counts.). It’s nice to know you are very unlikely to run into issues in the future.
