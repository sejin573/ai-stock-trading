from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
import zipfile


ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
OUTPUT_PATH = DOCS_DIR / "portfolio_dashboard_presentation.pptx"

SLIDES = [
    {
        "title": "뉴스 기반 한국 주식 자동 포트폴리오 대시보드",
        "subtitle": "Naver News + KIS Open API + pykrx + Streamlit",
        "bullets": [
            "한국 주식의 뉴스 흐름과 주가 변화를 함께 해석하는 포트폴리오용 대시보드",
            "실시간 가격 확인, 상승 기대주 탐색, 모의 자동매매를 하나의 흐름으로 연결",
        ],
    },
    {
        "title": "문제 정의",
        "subtitle": "왜 이 대시보드가 필요한가",
        "bullets": [
            "뉴스와 주가를 따로 보느라 단기 의사결정 속도가 느려짐",
            "변동성이 크고 기대 수익이 높은 종목을 빠르게 추리기 어려움",
            "보유 종목 손익, 매매 이력, 평가금액을 한 번에 보기 어려움",
        ],
    },
    {
        "title": "데이터 소스와 구조",
        "subtitle": "한국 시장 전용 데이터 조합",
        "bullets": [
            "네이버 뉴스 검색 API: 종목별 뉴스 수집",
            "KIS Open API: 실시간 시세, 모의투자 주문 및 잔고",
            "pykrx: 종목 전체 목록, 보조 시세, 백테스트 친화 데이터",
            "Streamlit: 투자 모니터링에 적합한 대시보드 UI",
        ],
    },
    {
        "title": "핵심 기능",
        "subtitle": "현재 구현 상태",
        "bullets": [
            "한국 상장사 전체 목록 기반 종목 선택",
            "3초 주기 실시간 갱신 차트와 포트폴리오 상태판",
            "변동성, 뉴스 감성, 기대 상승률 기반 추천 후보 탐색",
            "모의 자동매수/자동매도와 최근 거래 이력 표시",
        ],
    },
    {
        "title": "자동매매 및 학습 루프",
        "subtitle": "후보 탐색에서 포지션 관리까지",
        "bullets": [
            "시장 후보군 스캔 후 점수와 필터 기준으로 매수 후보 선정",
            "기대 수익률과 목표 수익률 기준으로 자동 매도 판단",
            "거래 결과를 기반으로 가중치를 조정하는 온라인 학습 구조 반영",
            "실전 투입 전 모의투자로 전략을 반복 검증 가능",
        ],
    },
    {
        "title": "최근 UI 개선 사항",
        "subtitle": "포트폴리오 모니터링 강화",
        "bullets": [
            "보유 종목별 총 매수금액, 현재 평가금액, 평가손익 표시",
            "평가손익과 수익률 색상 강조로 상태를 즉시 식별",
            "수익률 상위, 손실률 상위, 평가금액 상위 정렬 토글 제공",
            "총 투자원금, 총 평가금액, 누적 손익, 누적 수익률 카드 구성",
        ],
    },
    {
        "title": "기대 효과와 향후 확장",
        "subtitle": "포트폴리오 프로젝트의 성장 방향",
        "bullets": [
            "단기 유망주 탐색 속도 향상",
            "뉴스 이벤트와 가격 반응을 동시에 관찰하는 분석 환경 확보",
            "리스크 규칙, 포지션 사이징, 실거래 연계 기능으로 확장 가능",
            "학습 데이터 축적을 통한 추천 정밀도 개선 기대",
        ],
    },
]


def emu(value_in_inches: float) -> int:
    return int(value_in_inches * 914400)


def build_text_shape(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, paragraphs: list[tuple[str, int, bool]]) -> str:
    paragraph_xml = []
    for text, font_size, bold in paragraphs:
        escaped_text = escape(text)
        bold_attr = ' b="1"' if bold else ""
        paragraph_xml.append(
            f"""
        <a:p>
          <a:r>
            <a:rPr lang="ko-KR" sz="{font_size}"{bold_attr} dirty="0"/>
            <a:t>{escaped_text}</a:t>
          </a:r>
          <a:endParaRPr lang="ko-KR" sz="{font_size}"/>
        </a:p>"""
        )

    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="{escape(name)}"/>
        <p:cNvSpPr txBox="1"/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm>
          <a:off x="{x}" y="{y}"/>
          <a:ext cx="{cx}" cy="{cy}"/>
        </a:xfrm>
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" rtlCol="0" anchor="t"/>
        <a:lstStyle/>
        {''.join(paragraph_xml)}
      </p:txBody>
    </p:sp>"""


def build_slide_xml(title: str, subtitle: str, bullets: list[str]) -> str:
    title_shape = build_text_shape(
        shape_id=2,
        name="Title",
        x=emu(0.7),
        y=emu(0.45),
        cx=emu(11.0),
        cy=emu(0.8),
        paragraphs=[(title, 2800, True)],
    )
    subtitle_shape = build_text_shape(
        shape_id=3,
        name="Subtitle",
        x=emu(0.72),
        y=emu(1.32),
        cx=emu(10.8),
        cy=emu(0.5),
        paragraphs=[(subtitle, 1600, False)],
    )
    body_paragraphs = [(f"• {bullet}", 1800, False) for bullet in bullets]
    body_shape = build_text_shape(
        shape_id=4,
        name="Body",
        x=emu(0.9),
        y=emu(2.0),
        cx=emu(10.3),
        cy=emu(4.5),
        paragraphs=body_paragraphs,
    )

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg>
      <p:bgPr>
        <a:solidFill>
          <a:srgbClr val="F8FAFC"/>
        </a:solidFill>
        <a:effectLst/>
      </p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      {title_shape}
      {subtitle_shape}
      {body_shape}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr>
    <a:masterClrMapping/>
  </p:clrMapOvr>
</p:sld>
"""


def build_slide_rel_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>
"""


def build_presentation_xml(slide_count: int) -> str:
    slide_ids = []
    for index in range(slide_count):
        slide_ids.append(f'    <p:sldId id="{256 + index}" r:id="rId{index + 2}"/>')
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                saveSubsetFonts="1"
                autoCompressPictures="0">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rId1"/>
  </p:sldMasterIdLst>
  <p:sldIdLst>
{chr(10).join(slide_ids)}
  </p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>
"""


def build_presentation_rels_xml(slide_count: int) -> str:
    relationships = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    ]
    for index in range(slide_count):
        relationships.append(
            f'<Relationship Id="rId{index + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{index + 1}.xml"/>'
        )
    relationships.extend(
        [
            f'<Relationship Id="rId{slide_count + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>',
            f'<Relationship Id="rId{slide_count + 3}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>',
            f'<Relationship Id="rId{slide_count + 4}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles" Target="tableStyles.xml"/>',
        ]
    )
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  """ + "\n  ".join(relationships) + """
</Relationships>
"""


def build_content_types_xml(slide_count: int) -> str:
    slide_overrides = []
    for index in range(slide_count):
        slide_overrides.append(
            f'<Override PartName="/ppt/slides/slide{index + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        )
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  """ + "\n  ".join(slide_overrides) + """
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>
  <Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>
  <Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>
</Types>
"""


def build_root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def build_app_xml(slide_count: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>{slide_count}</Slides>
  <Notes>0</Notes>
  <HiddenSlides>0</HiddenSlides>
  <MMClips>0</MMClips>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant>
        <vt:lpstr>Theme</vt:lpstr>
      </vt:variant>
      <vt:variant>
        <vt:i4>1</vt:i4>
      </vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="1" baseType="lpstr">
      <vt:lpstr>Office Theme</vt:lpstr>
    </vt:vector>
  </TitlesOfParts>
  <Company>OpenAI Codex</Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>1.0</AppVersion>
</Properties>
"""


def build_core_xml() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>뉴스 기반 한국 주식 자동 포트폴리오 대시보드</dc:title>
  <dc:subject>Portfolio Presentation</dc:subject>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:keywords>한국주식, 자동매매, 포트폴리오, 뉴스분석</cp:keywords>
  <dc:description>포트폴리오 프로젝트 발표용 PPT</dc:description>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>
"""


def build_pres_props_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentationPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                  xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>
"""


def build_view_props_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:viewPr xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
          xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:normalViewPr>
    <p:restoredLeft sz="15620"/>
    <p:restoredTop sz="94660"/>
  </p:normalViewPr>
  <p:slideViewPr>
    <p:cSldViewPr snapToGrid="1" snapToObjects="1" showGuides="1"/>
  </p:slideViewPr>
  <p:notesTextViewPr/>
  <p:gridSpacing cx="780288" cy="780288"/>
</p:viewPr>
"""


def build_table_styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>
"""


def build_theme_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Codex Theme">
  <a:themeElements>
    <a:clrScheme name="Codex">
      <a:dk1><a:srgbClr val="0F172A"/></a:dk1>
      <a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1E293B"/></a:dk2>
      <a:lt2><a:srgbClr val="E2E8F0"/></a:lt2>
      <a:accent1><a:srgbClr val="0F766E"/></a:accent1>
      <a:accent2><a:srgbClr val="2563EB"/></a:accent2>
      <a:accent3><a:srgbClr val="16A34A"/></a:accent3>
      <a:accent4><a:srgbClr val="EA580C"/></a:accent4>
      <a:accent5><a:srgbClr val="B91C1C"/></a:accent5>
      <a:accent6><a:srgbClr val="7C3AED"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink>
      <a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Codex Fonts">
      <a:majorFont>
        <a:latin typeface="Aptos Display"/>
        <a:ea typeface="Malgun Gothic"/>
        <a:cs typeface="Arial"/>
      </a:majorFont>
      <a:minorFont>
        <a:latin typeface="Aptos"/>
        <a:ea typeface="Malgun Gothic"/>
        <a:cs typeface="Arial"/>
      </a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="Codex Format">
      <a:fillStyleLst>
        <a:solidFill><a:schemeClr val="lt1"/></a:solidFill>
        <a:solidFill><a:schemeClr val="accent1"/></a:solidFill>
        <a:solidFill><a:schemeClr val="accent2"/></a:solidFill>
      </a:fillStyleLst>
      <a:lnStyleLst>
        <a:ln w="9525" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="accent1"/></a:solidFill></a:ln>
        <a:ln w="25400" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="accent2"/></a:solidFill></a:ln>
        <a:ln w="38100" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:schemeClr val="accent3"/></a:solidFill></a:ln>
      </a:lnStyleLst>
      <a:effectStyleLst>
        <a:effectStyle><a:effectLst/></a:effectStyle>
        <a:effectStyle><a:effectLst/></a:effectStyle>
        <a:effectStyle><a:effectLst/></a:effectStyle>
      </a:effectStyleLst>
      <a:bgFillStyleLst>
        <a:solidFill><a:schemeClr val="lt1"/></a:solidFill>
        <a:solidFill><a:schemeClr val="lt2"/></a:solidFill>
        <a:solidFill><a:schemeClr val="dk1"/></a:solidFill>
      </a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/>
  <a:extraClrSchemeLst/>
</a:theme>
"""


def build_slide_master_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld name="Codex Slide Master">
    <p:bg>
      <p:bgPr>
        <a:solidFill>
          <a:srgbClr val="F8FAFC"/>
        </a:solidFill>
        <a:effectLst/>
      </p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst>
    <p:sldLayoutId id="2147483649" r:id="rId1"/>
  </p:sldLayoutIdLst>
  <p:txStyles>
    <p:titleStyle>
      <a:lvl1pPr algn="l">
        <a:defRPr sz="2800" b="1"/>
      </a:lvl1pPr>
    </p:titleStyle>
    <p:bodyStyle>
      <a:lvl1pPr marL="0" indent="0">
        <a:defRPr sz="1800"/>
      </a:lvl1pPr>
    </p:bodyStyle>
    <p:otherStyle/>
  </p:txStyles>
</p:sldMaster>
"""


def build_slide_master_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
"""


def build_slide_layout_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             type="titleAndContent"
             preserve="1">
  <p:cSld name="Title and Content">
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr>
    <a:masterClrMapping/>
  </p:clrMapOvr>
</p:sldLayout>
"""


def build_slide_layout_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
"""


def write_text(zf: zipfile.ZipFile, path: str, content: str) -> None:
    zf.writestr(path, content.encode("utf-8"))


def build_pptx(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        write_text(zf, "[Content_Types].xml", build_content_types_xml(len(SLIDES)))
        write_text(zf, "_rels/.rels", build_root_rels_xml())
        write_text(zf, "docProps/app.xml", build_app_xml(len(SLIDES)))
        write_text(zf, "docProps/core.xml", build_core_xml())
        write_text(zf, "ppt/presentation.xml", build_presentation_xml(len(SLIDES)))
        write_text(zf, "ppt/_rels/presentation.xml.rels", build_presentation_rels_xml(len(SLIDES)))
        write_text(zf, "ppt/presProps.xml", build_pres_props_xml())
        write_text(zf, "ppt/viewProps.xml", build_view_props_xml())
        write_text(zf, "ppt/tableStyles.xml", build_table_styles_xml())
        write_text(zf, "ppt/theme/theme1.xml", build_theme_xml())
        write_text(zf, "ppt/slideMasters/slideMaster1.xml", build_slide_master_xml())
        write_text(zf, "ppt/slideMasters/_rels/slideMaster1.xml.rels", build_slide_master_rels_xml())
        write_text(zf, "ppt/slideLayouts/slideLayout1.xml", build_slide_layout_xml())
        write_text(zf, "ppt/slideLayouts/_rels/slideLayout1.xml.rels", build_slide_layout_rels_xml())

        for index, slide in enumerate(SLIDES, start=1):
            write_text(
                zf,
                f"ppt/slides/slide{index}.xml",
                build_slide_xml(slide["title"], slide["subtitle"], slide["bullets"]),
            )
            write_text(zf, f"ppt/slides/_rels/slide{index}.xml.rels", build_slide_rel_xml())


def main() -> None:
    build_pptx(OUTPUT_PATH)
    print(f"created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
