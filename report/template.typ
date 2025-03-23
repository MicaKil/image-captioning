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
	set heading(numbering: (..nums) => {
		let numbers = nums.pos()
		if numbers.len() <= 3 {
			numbering("1.", ..numbers)
		}
	})

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
		// first-line-indent: 1.25cm
	)

	doc
}