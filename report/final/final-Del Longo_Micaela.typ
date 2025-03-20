#import "../template.typ": conf

#set document(
	title: "Trabajo Final: Descripción de Imágenes con Mecanismo de Atención",
	author: "Micaela Del Longo",
	date: datetime(year: 2025, month: 3, day: 27)
)

#show: conf.with(
	title: [#context(document.title)],
	author: [#context(document.author.first())],
	date: [#context(document.date.display("[day] de marzo de [year]"))],
	university: [Universidad Nacional de Cuyo],
	college: [Facultad de Ingeniería],
	career: [Licenciatura en Ciencias de la Computación],
	course: [Inteligencia Artificial II],
	professors: ([Dr. Rodrigo Gonzalez], [Dr. Jorge Guerra]),
)

#outline(
	title: "Contenidos",
	indent: auto
)

= Introducción

