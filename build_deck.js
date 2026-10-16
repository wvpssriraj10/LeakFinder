// Usage: node build_deck.js deck.json out.pptx   (JSON -> PPTX, native charts/tables)
const pptxgen = require("pptxgenjs"); const fs = require("fs");
const J = JSON.parse(fs.readFileSync(process.argv[2] || "deck.json")); const OUT = process.argv[3] || "LeakFinder_Round1.pptx";
const p = new pptxgen(); p.layout = "LAYOUT_16x9"; p.title = J.meta.title;
const NAVY="0F172A", TEAL="14B8A6", INK="1E293B", MUTE="64748B", BG="F8FAFC", WHITE="FFFFFF", AMB="F59E0B", RED="EF4444", IND="6366F1", F="Calibri";
const ACC=[TEAL,AMB,RED,IND];
const tb=(s,t,o)=>s.addText(t,Object.assign({fontFace:F,margin:0,isTextBox:true},o));
function base(sl){ const s=p.addSlide(); s.background={color:BG};
  s.addShape(p.ShapeType.rect,{x:0,y:0,w:0.18,h:5.625,fill:{color:TEAL},line:{color:TEAL}});
  tb(s,sl.title,{x:0.55,y:0.28,w:8.9,h:0.6,fontSize:26,bold:true,color:NAVY});
  if(sl.subtitle) tb(s,sl.subtitle,{x:0.55,y:0.88,w:8.9,h:0.32,fontSize:13,color:MUTE});
  tb(s,`${J.meta.event}  |  ${J.meta.team}  |  ${sl.id}`,{x:0.55,y:5.25,w:8.9,h:0.22,fontSize:9,color:MUTE}); return s; }
function card(s,x,y,w,h,head,body,ac,fs){ s.addShape(p.ShapeType.rect,{x,y,w,h,fill:{color:WHITE},line:{color:"E2E8F0",width:1}});
  s.addShape(p.ShapeType.rect,{x,y,w:0.07,h,fill:{color:ac},line:{color:ac}});
  tb(s,head,{x:x+0.2,y:y+0.1,w:w-0.3,h:0.32,fontSize:13,bold:true,color:NAVY});
  const items=(Array.isArray(body)?body:[body]).map((t,i,a)=>({text:t,options:{bullet:Array.isArray(body),breakLine:i<a.length-1,paraSpaceAfter:5}}));
  tb(s,items,{x:x+0.2,y:y+0.5,w:w-0.35,h:h-0.58,fontSize:fs||11.5,color:INK,valign:"top"}); }
function chart(s,c,x,y,w,h){ s.addChart(p.charts.BAR,[{name:c.title,labels:c.labels,values:c.values}],{x,y,w,h,barDir:"bar",chartColors:[TEAL],showTitle:true,title:c.title,titleFontSize:12,titleColor:NAVY,
  showValue:true,dataLabelFontSize:9,dataLabelColor:INK,dataLabelFormatCode:"#,##0",catAxisLabelColor:INK,catAxisLabelFontSize:10,valAxisLabelColor:MUTE,valAxisLabelFontSize:9,
  valGridLine:{color:"E2E8F0",size:0.5},catGridLine:{style:"none"},showLegend:false,catAxisOrientation:"maxMin"}); }
function table(s,t,x,y,w,colW,fs){ const hdr=t.columns.map(c=>({text:c,options:{bold:true,color:WHITE,fill:{color:NAVY},fontFace:F,fontSize:fs,valign:"middle"}}));
  const rows=t.rows.map((r,i)=>r.map(c=>({text:String(c),options:{color:INK,fill:{color:i%2?"F1F5F9":WHITE},fontFace:F,fontSize:fs,valign:"middle"}})));
  s.addTable([hdr,...rows],{x,y,w,colW,border:{type:"solid",color:"E2E8F0",pt:0.75},margin:[0.05,0.08,0.05,0.08]}); }
for(const sl of J.slides){
  if(sl.type==="title"){ const s=p.addSlide(); s.background={color:NAVY};
    s.addShape(p.ShapeType.rect,{x:0.6,y:1.2,w:0.9,h:0.07,fill:{color:TEAL},line:{color:TEAL}});
    tb(s,sl.title,{x:0.6,y:1.4,w:8.8,h:0.9,fontSize:46,bold:true,color:WHITE}); tb(s,sl.subtitle,{x:0.6,y:2.4,w:8.4,h:0.8,fontSize:19,color:"CBD5E1"});
    tb(s,sl.meta,{x:0.6,y:4.4,w:8.8,h:0.3,fontSize:12,color:TEAL}); tb(s,"Data Science Club x IIEC  |  VIT-AP University",{x:0.6,y:4.8,w:8.8,h:0.3,fontSize:11,color:"94A3B8"}); continue; }
  const s=base(sl);
  if(sl.type==="bullets_cards"){ const n=sl.cards.length, gap=0.15, w=(8.9-gap*(n-1))/n; sl.cards.forEach((c,i)=>card(s,0.55+i*(w+gap),1.45,w,3.5,c.head,c.body,ACC[i%4])); }
  if(sl.type==="stats_chart"){ sl.stats.forEach((st,i)=>{ const x=0.55+(i%2)*2.55, y=1.45+Math.floor(i/2)*1.7;
      s.addShape(p.ShapeType.rect,{x,y,w:2.4,h:1.55,fill:{color:WHITE},line:{color:"E2E8F0",width:1}}); s.addShape(p.ShapeType.rect,{x,y,w:0.07,h:1.55,fill:{color:ACC[i]},line:{color:ACC[i]}});
      tb(s,st.value,{x:x+0.2,y:y+0.2,w:2.1,h:0.6,fontSize:26,bold:true,color:NAVY}); tb(s,st.label,{x:x+0.2,y:y+0.85,w:2.1,h:0.6,fontSize:11,color:MUTE,valign:"top"}); });
    chart(s,sl.chart,5.75,1.4,3.75,3.6); }
  if(sl.type==="usp"){ s.addShape(p.ShapeType.rect,{x:0.55,y:1.35,w:8.9,h:1.15,fill:{color:NAVY},line:{color:NAVY}});
    s.addText([{text:"USP  ",options:{bold:true,color:TEAL}},{text:sl.usp,options:{color:WHITE}}],{x:0.75,y:1.4,w:8.5,h:1.05,fontFace:F,fontSize:16,margin:0,valign:"middle",isTextBox:true});
    const n=sl.cards.length,gap=0.15,w=(8.9-gap*(n-1))/n; sl.cards.forEach((c,i)=>card(s,0.55+i*(w+gap),2.7,w,2.35,c.head,c.body,ACC[i%4],11)); }
  if(sl.type==="mvp"){ sl.steps.forEach((t,i)=>{ const x=0.55+i*3.05; s.addShape(p.ShapeType.rect,{x,y:1.45,w:2.8,h:2.35,fill:{color:WHITE},line:{color:"E2E8F0",width:1}});
      s.addShape(p.ShapeType.ellipse,{x:x+0.15,y:1.6,w:0.5,h:0.5,fill:{color:TEAL},line:{color:TEAL}}); tb(s,t.n,{x:x+0.15,y:1.6,w:0.5,h:0.5,fontSize:16,bold:true,color:WHITE,align:"center",valign:"middle"});
      tb(s,t.head,{x:x+0.8,y:1.6,w:1.9,h:0.5,fontSize:15,bold:true,color:NAVY,valign:"middle"}); tb(s,t.body,{x:x+0.2,y:2.25,w:2.45,h:1.45,fontSize:11.5,color:INK,valign:"top"}); });
    tb(s,sl.demo,{x:0.55,y:4.05,w:8.9,h:0.8,fontSize:13,color:NAVY,bold:true,valign:"top"}); }
  if(sl.type==="chart_insights"){ chart(s,sl.chart,0.55,1.4,5.0,3.65);
    sl.insights.forEach((c,i)=>card(s,5.75,1.4+i*1.25,3.75,1.15,c.head,c.body,ACC[i%4],10.5)); }
  if(sl.type==="table"){ table(s,sl.table,0.55,1.5,8.9,[3.2,1.0,1.2,2.0,1.5],12);
    s.addShape(p.ShapeType.rect,{x:0.55,y:3.55,w:8.9,h:1.4,fill:{color:WHITE},line:{color:"E2E8F0",width:1}}); s.addShape(p.ShapeType.rect,{x:0.55,y:3.55,w:0.07,h:1.4,fill:{color:AMB},line:{color:AMB}});
    tb(s,sl.takeaway,{x:0.8,y:3.65,w:8.5,h:1.2,fontSize:13,color:INK,valign:"middle"}); }
  if(sl.type==="actions"){ table(s,sl.table,0.55,1.45,8.9,[0.5,5.2,1.7,1.5],12); tb(s,sl.takeaway,{x:0.55,y:4.75,w:8.9,h:0.35,fontSize:11,color:MUTE,italic:true}); }
}
p.writeFile({fileName:OUT}).then(()=>console.log("wrote",OUT));
