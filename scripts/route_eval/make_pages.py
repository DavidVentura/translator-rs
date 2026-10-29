"""Render the OCR script-routing eval pages: phone-sized screenshots of text whose
script mix is known, plus a manifest naming the non-Latin recognizer scripts each
page contains (the labels `route_eval` checks routing against).

    python3 scripts/route_eval/make_pages.py <out_dir>
"""

import subprocess
import sys
from pathlib import Path

STYLE = """
body { font-family: sans-serif; font-size: 19px; line-height: 1.5; padding: 14px; margin: 0;
       background: #fff; color: #222; }
h2 { font-size: 24px; } li { margin: 12px 0; }
"""

EN = """<h2>Weekend plans</h2>
<p>The museum opens at ten on Saturdays, so we should leave the house early and grab coffee on the way.
After lunch we can walk along the river and visit the old bookshop near the bridge.</p>
<p>If it rains, the cinema downtown is showing a documentary about volcanoes. Tickets are cheaper before
five in the afternoon, and students get an extra discount with a valid card.</p>
<ul><li><b>Bring:</b> umbrella, water, a charged phone</li><li><b>Budget:</b> about forty dollars each</li></ul>"""

FR = """<h2>Recette de la soupe à l'oignon</h2>
<p>Émincez finement les oignons et faites-les revenir doucement dans le beurre pendant une demi-heure,
jusqu'à ce qu'ils soient bien dorés. Ajoutez une cuillère de farine et mélangez.</p>
<p>Versez le bouillon chaud, salez, poivrez et laissez mijoter vingt minutes. Servez avec des tranches de
pain grillé et beaucoup de fromage râpé, puis passez le tout sous le gril.</p>"""

DE = """<h2>Hinweise für Besucher</h2>
<p>Bitte melden Sie sich beim Empfang an und tragen Sie Ihren Besucherausweis jederzeit sichtbar.
Das Fotografieren in den Laborräumen ist aus Sicherheitsgründen nicht gestattet.</p>
<p>Im Brandfall verlassen Sie das Gebäude über die gekennzeichneten Notausgänge und folgen den
Anweisungen des Personals. Der Sammelplatz befindet sich auf dem Parkplatz gegenüber.</p>"""

RU = """<h2>Правила для посетителей</h2>
<p>Пожалуйста, зарегистрируйтесь на стойке информации и носите пропуск на видном месте.
Фотографировать в лабораториях запрещено по соображениям безопасности.</p>
<p>При пожаре покиньте здание через обозначенные аварийные выходы и следуйте указаниям персонала.
Место сбора находится на парковке напротив главного входа.</p>"""

RU_TERMS = """<h2>Поддержка клиентов</h2>
<p>Если приложение не запускается на iPhone или Android, сначала обновите его до последней версии
в App Store или Google Play. Затем перезагрузите телефон.</p>
<p>Напишите нам на support@example.com или позвоните Ивану Петрову по номеру +7 495 123-45-67.
Мы отвечаем в течение 24 часов, кроме выходных. Укажите модель устройства, например Samsung Galaxy S23.</p>"""

RU_EN_GLOSS = """<ul>
<li><b>Past Tense &amp; Cases:</b> Они сказали тебе результаты <b>экзамена после</b> того, как банк закрылся. (They told you the results of the exam after the bank closed.)</li>
<li><b>Idiomatic / Tricky Expression: У природы</b> нет плохой погоды. (There is no bad weather in nature / Nature knows no bad weather — a famous poetic idiom.)</li>
<li><b>Reflexive / Impersonal:</b> Мне хотелось бы узнать, во сколько открывается <b>музей.</b> (I would like to know what time the museum opens.)</li>
<li><b>Aspect:</b> Я читал эту книгу всё лето, но так и не дочитал. (I was reading this book all summer but never finished it.)</li></ul>"""

RU_TRANSLIT_FR = """<p>Here are sample sentences paired in Russian and French:</p><ul>
<li><b>Он усердно работает</b> <i>(On userdno rabotayet)</i> – <b>Il travaille dur</b></li>
<li><b>Солнце яркое</b> <i>(Solntse yarkoye)</i> – <b>Le soleil est brillant</b></li>
<li><b>Я завтракаю</b> <i>(Ya zavtrakayu)</i> – <b>Je prends le petit-déjeuner</b></li>
<li><b>Она пьёт воду</b> <i>(Ona pyot vodu)</i> – <b>Elle boit de l'eau</b></li>
<li><b>Мы пишем письма</b> <i>(My pishem pisma)</i> – <b>Nous écrivons des lettres</b></li>
<li><b>Я иду в парк</b> <i>(Ya idu v park)</i> – <b>Je marche au parc</b></li></ul>"""

EN_RU_WORDS = """<h2>Travel notes</h2>
<p>On the second day we took the train to Kazan and stayed near the river. Our host, Мария, kept
recommending a dish called <b>эчпочмак</b>, which turned out to be a triangular pastry with meat and potato.</p>
<p>The station signs were mostly in Russian, so the word <b>выход</b> (exit) became very familiar. The
metro is cheap, clean and runs every few minutes even late in the evening.</p>"""

FR_RU_COLUMNS = """<table style="width:100%;border-collapse:collapse"><tr>
<td style="width:50%;vertical-align:top;padding-right:8px">Bienvenue au musée. Les billets sont en vente à l'accueil. Les enfants de moins de six ans entrent gratuitement.</td>
<td style="width:50%;vertical-align:top;padding-left:8px">Добро пожаловать в музей. Билеты продаются у входа. Дети до шести лет проходят бесплатно.</td>
</tr><tr>
<td style="vertical-align:top;padding-right:8px">Il est interdit de toucher les œuvres et d'utiliser le flash.</td>
<td style="vertical-align:top;padding-left:8px">Запрещается трогать экспонаты и пользоваться вспышкой.</td>
</tr></table>"""

EN_RU_QUOTE = """<h2>A letter from Moscow</h2>
<p>My grandmother kept every letter her sister sent from Moscow in a tin box under the bed. Most of them
are about the weather, the price of bread and which neighbours had moved away. The handwriting is neat
and the paper has turned yellow at the edges.</p>
<p>One letter from the winter of 1987 ends with a line she underlined twice:</p>
<p><b>Береги себя и пиши почаще.</b></p>
<p>It means "take care of yourself and write more often". We framed it and hung it in the kitchen, next
to the photograph of the two sisters standing on the beach in Sochi when they were still girls.</p>"""

HI = """<h2>आगंतुकों के लिए नियम</h2>
<p>कृपया स्वागत कक्ष पर पंजीकरण करें और अपना पास हर समय दिखाई देने वाली जगह पर पहनें।
सुरक्षा कारणों से प्रयोगशालाओं में फोटो लेना मना है।</p>
<p>आग लगने की स्थिति में चिह्नित आपातकालीन निकास से बाहर निकलें और कर्मचारियों के निर्देशों का पालन करें।</p>"""

AR = """<h2 dir="rtl">تعليمات للزوار</h2>
<p dir="rtl">يرجى التسجيل في مكتب الاستقبال وارتداء بطاقة الزائر في مكان ظاهر طوال الوقت.
يمنع التصوير داخل المختبرات لأسباب تتعلق بالسلامة.</p>
<p dir="rtl">في حالة الحريق، غادر المبنى عبر مخارج الطوارئ المحددة واتبع تعليمات الموظفين.</p>"""

# (name, html, non-Latin recognizer scripts present on the page)
PAGES = [
    ("en", EN, []),
    ("fr", FR, []),
    ("de", DE, []),
    ("ru", RU, ["eslav"]),
    ("ru_terms", RU_TERMS, ["eslav"]),
    ("ru_en_gloss", RU_EN_GLOSS, ["eslav"]),
    ("ru_translit_fr", RU_TRANSLIT_FR, ["eslav"]),
    ("en_ru_words", EN_RU_WORDS, ["eslav"]),
    ("en_ru_quote", EN_RU_QUOTE, ["eslav"]),
    ("fr_ru_columns", FR_RU_COLUMNS, ["eslav"]),
    ("hi", HI, ["devanagari"]),
    ("ar", AR, ["arabic"]),
]


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, body, scripts in PAGES:
        html = out / f"{name}.html"
        html.write_text(
            f'<!doctype html><html><head><meta charset="utf-8"><style>{STYLE}</style></head>'
            f"<body>{body}</body></html>"
        )
        png = out / f"{name}.png"
        subprocess.run(
            [
                "chromium",
                "--headless",
                "--hide-scrollbars",
                "--force-device-scale-factor=2.625",
                "--window-size=412,915",
                f"--screenshot={png}",
                html.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
        manifest.append(f"{png.name}\t{','.join(scripts)}")
    (out / "manifest.tsv").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
