"""Thư viện nhỏ sinh file .drawio (mxGraph XML) — mở trực tiếp ở https://app.diagrams.net.

Dùng bởi scripts/make_drawio.py. Mỗi sơ đồ = 1 file .drawio riêng để dễ sửa.
"""
from html import escape
from pathlib import Path

# ---- Bảng màu dùng chung cho MỌI sơ đồ (đổi ở đây là đổi hết) ----
PAL = {
    'input':    ('#E3F2FD', '#1565C0', '#0D47A1'),   # dữ liệu vào / ảnh
    'stem':     ('#E0F7FA', '#00838F', '#006064'),   # stem
    'backbone': ('#E8F5E9', '#2E7D32', '#1B5E20'),   # backbone
    'attn':     ('#FCE4EC', '#AD1457', '#880E4F'),   # attention
    'neck':     ('#FFF3E0', '#EF6C00', '#E65100'),   # neck / FPN
    'head':     ('#F3E5F5', '#6A1B9A', '#4A148C'),   # head
    'out':      ('#ECEFF1', '#455A64', '#263238'),   # đầu ra / decode
    'train':    ('#FFFDE7', '#F9A825', '#F57F17'),   # chỉ có lúc train
    'deploy':   ('#EDE7F6', '#4527A0', '#311B92'),   # export / deploy
    'op':       ('#FFFFFF', '#90A4AE', '#37474F'),   # phép toán (+, upsample)
    'note':     ('#FAFAFA', '#BDBDBD', '#616161'),   # ghi chú
    'bad':      ('#FFEBEE', '#C62828', '#B71C1C'),   # cũ / bị bỏ
    'good':     ('#E8F5E9', '#2E7D32', '#1B5E20'),
}

FONT = 'Helvetica'


def _esc(s):
    return escape(str(s), quote=True)


class Drawio:
    def __init__(self, name, page_w=1600, page_h=1200):
        self.name = name
        self.page_w, self.page_h = page_w, page_h
        self.cells = []
        self._n = 0

    def _id(self, prefix='n'):
        self._n += 1
        return f'{prefix}{self._n}'

    # ---------- vertex ----------
    def box(self, x, y, w, h, label, kind='op', *, rounded=1, dashed=0, fontsize=12,
            align='center', bold_first=True, parent='1', extra='', fill=None, stroke=None,
            fontcolor=None, opacity=100):
        f, s, fc = PAL[kind]
        style = (f'rounded={rounded};whiteSpace=wrap;html=1;'
                 f'fillColor={fill or f};strokeColor={stroke or s};fontColor={fontcolor or fc};'
                 f'strokeWidth=1.5;fontSize={fontsize};fontFamily={FONT};align={align};'
                 f'verticalAlign=middle;arcSize=12;dashed={dashed};opacity={opacity};{extra}')
        cid = self._id()
        self.cells.append(
            f'<mxCell id="{cid}" value="{_esc(label)}" style="{_esc(style)}" vertex="1" parent="{parent}">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def ellipse(self, x, y, w, h, label, kind='op', fontsize=14, extra=''):
        f, s, fc = PAL[kind]
        style = (f'ellipse;whiteSpace=wrap;html=1;fillColor={f};strokeColor={s};fontColor={fc};'
                 f'strokeWidth=1.5;fontSize={fontsize};fontFamily={FONT};{extra}')
        cid = self._id()
        self.cells.append(
            f'<mxCell id="{cid}" value="{_esc(label)}" style="{_esc(style)}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def group_frame(self, x, y, w, h, label, kind='note', dashed=1, fontsize=13, fill=None):
        """Khung bao (vẽ dưới cùng — gọi TRƯỚC các box bên trong)."""
        f, s, fc = PAL[kind]
        style = (f'rounded=1;whiteSpace=wrap;html=1;fillColor={fill or f};strokeColor={s};'
                 f'fontColor={fc};dashed={dashed};dashPattern=8 6;strokeWidth=1.5;'
                 f'fontSize={fontsize};fontFamily={FONT};verticalAlign=top;align=left;'
                 f'spacingLeft=12;spacingTop=6;arcSize=6;fontStyle=1')
        cid = self._id('g')
        self.cells.append(
            f'<mxCell id="{cid}" value="{_esc(label)}" style="{_esc(style)}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def text(self, x, y, w, h, label, fontsize=12, color='#455A64', align='left', bold=False):
        style = (f'text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign=top;'
                 f'whiteSpace=wrap;fontSize={fontsize};fontFamily={FONT};fontColor={color};'
                 f'fontStyle={1 if bold else 0}')
        cid = self._id('t')
        self.cells.append(
            f'<mxCell id="{cid}" value="{_esc(label)}" style="{_esc(style)}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def title(self, x, y, w, label, sub=None):
        self.text(x, y, w, 34, label, fontsize=22, color='#102A43', bold=True)
        if sub:
            self.text(x, y + 30, w, 40, sub, fontsize=13, color='#627D98')

    # ---------- edge ----------
    def edge(self, src, dst, label='', *, style='ortho', color='#37474F', dashed=0,
             width=2, exit_=None, entry=None, fontsize=11, waypoints=None, arrow='classic',
             start_arrow='none', label_x=None, curved=0):
        base = {
            'ortho': 'edgeStyle=orthogonalEdgeStyle;rounded=1;',
            'straight': 'edgeStyle=none;rounded=0;',
            'elbow': 'edgeStyle=elbowEdgeStyle;rounded=1;',
        }[style]
        s = (base + f'html=1;strokeColor={color};strokeWidth={width};dashed={dashed};'
             f'fontSize={fontsize};fontFamily={FONT};fontColor={color};'
             f'endArrow={arrow};startArrow={start_arrow};curved={curved};'
             f'labelBackgroundColor=#FFFFFF;jettySize=auto;orthogonalLoop=1;')
        if exit_:
            ex, ey = exit_
            s += f'exitX={ex};exitY={ey};exitDx=0;exitDy=0;'
        if entry:
            ix, iy = entry
            s += f'entryX={ix};entryY={iy};entryDx=0;entryDy=0;'
        cid = self._id('e')
        geo = '<mxGeometry relative="1" as="geometry">'
        if waypoints:
            geo += '<Array as="points">' + ''.join(
                f'<mxPoint x="{p[0]}" y="{p[1]}"/>' for p in waypoints) + '</Array>'
        geo += '</mxGeometry>'
        self.cells.append(
            f'<mxCell id="{cid}" value="{_esc(label)}" style="{_esc(s)}" edge="1" parent="1" '
            f'source="{src}" target="{dst}">{geo}</mxCell>')
        return cid

    # ---------- legend ----------
    def legend(self, x, y, items, w=230, title='Chú giải'):
        h = 34 + 26 * len(items)
        self.group_frame(x, y, w, h, title, kind='note', dashed=0, fontsize=12, fill='#FFFFFF')
        for i, (kind, label) in enumerate(items):
            f, s, _ = PAL[kind]
            yy = y + 30 + 26 * i
            self.box(x + 12, yy, 18, 16, '', kind, fontsize=1)
            self.text(x + 38, yy - 1, w - 50, 20, label, fontsize=11, color='#37474F')

    def save(self, path):
        body = '\n        '.join(self.cells)
        xml = f'''<mxfile host="app.diagrams.net" agent="pfdet-drone-follow" type="device">
  <diagram id="{self.name.replace(' ', '_')[:40]}" name="{_esc(self.name)}">
    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1"
        arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.page_w}" pageHeight="{self.page_h}"
        math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
        Path(path).write_text(xml, encoding='utf-8')
        return path
