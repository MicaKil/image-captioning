#import table: cell, header
#import "@preview/codly:1.2.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()

#let sub_caption(body) = {
  set text(size: 8pt)
  body
}

#let appendix(body) = {
  set heading(numbering: "A.1.", supplement: [Anexo])
  counter(heading).update(0)
  body
}

#let conf(
  title: none,
  author: (),
  date: datetime.today(),
  university: none,
  college: none,
  career: none,
  course: none,
  professors: (),
  doc,
) = {

  // TEXT
  set text(
    lang: "es",
    region: "ar",
    font: "Lora",
    // fallback: false,
  )
  
  // HEADING 
  // set heading(numbering: (..nums) => {
  //   let numbers = nums.pos()
  //   if numbers.len() <= 3 {
  //     numbering("1.1.", ..numbers)
  //   }
  // })
  set heading(numbering: "1.1.1.")
  show heading.where(level: 1): it => {
    set text(size: 14pt, weight: "medium")
    v(0.2em)
    it
    v(0.8em)
  }
  show heading.where(level: 2): it => {
    set text(size: 12pt, weight: "medium")
    it
    v(0.5em)
  }
  show heading.where(level: 3): it => {
    set text(size: 11pt, weight: "medium")
    it
    v(0.5em)
  }
  show heading.where(level: 4): it => {
    set text(size: 11pt, weight: "regular", style: "italic", fill: luma(130))
    it
  }
  
  
  // HEADER AND FOOTER
  set page(
    header: context {
      if counter(page).get().first() > 1 [
        #text(size: 10pt, fill: luma(130), title)
        #h(1fr)
        #text(size: 10pt, fill: luma(130), author)
      ]
    },
    header-ascent: 50%,
    footer: context {
      if counter(page).get().first() > 1 [
        #align(center)[
          #text(size: 10pt, fill: luma(130), "Página " + counter(page).display("1 de 1", both: true))
        ]
      ]
    },
    footer-descent: 50%,
    margin: 2cm //(x: 2cm, y: 1.5cm),
  )
  
  
  // COVER
  context {
    if counter(page).get().first() == 1 [
      #set page(
        margin: (y: 4cm),
        header: image("header-ingeniería.png", width: 65%),
        header-ascent: 50%
        )
      #set align(center + horizon)
      #set text(size: 14pt)
      #block(
        text(title, size: 16pt) + "\n" + author
      )
  
  
      #v(28pt)
  
      #text(college + ", " + university + "\n" + course + ", " + career)
  
      #v(28pt)
  
      #text(professors.join("\n"))
  
      #v(28pt)
  
      #date
    ]
  }
  
  
   // CONTENT
  
  pagebreak()
  
  set par(
    justify: true,
    linebreaks: "optimized",
    first-line-indent: (amount: 1.25cm, all: true)
  )
  
  // TABLE
  show table.cell.where(y: 0): it => {
    set text(weight: "semibold")
    it
  }
  
  set table(
    // fill: (x, y) =>
    //   if  y == 0 {
    //     gray.lighten(60%)
    //   },
    stroke: (x, y) => if y == 0 {
    (bottom: 0.7pt + black)
    },
    align: (x, y) => (
      if x > 0 { center }
      else { left }
    )
  )
  
  // LINKS
  show link: underline
  show link: set text(fill: blue)
  
  // show cite: underline
  show cite: c => [#set text(fill: blue);#super[#c]]  // has to be on the same line or it adds a space between text and citation.

  // show ref.where()
  // show ref: it => {
  //   let h = heading.supplement
  //   let el = it.element
  //   if el != none and el.func() == h {
  //     // Override equation references.
  //     link(el.location(),numbering(
  //       el.numbering,
  //       ..counter(eq).at(el.location())
  //     ))
  //   } else {
  //     // Other references as usual.
  //     it
  //   }
  // }
  // EQUATIONS
  set math.equation(numbering: "(1)")

  //CAPTIONS
  show figure.caption: set text(style: "italic")
  show figure.caption: c => [
    #text(weight: "semibold")[
      #c.supplement #context{c.counter.display(c.numbering)}#c.separator
    ]
    #c.body
  ]

  // LIST
  show list: it => {
    pad(left: 1.25cm)[#it]
  }

  doc
}