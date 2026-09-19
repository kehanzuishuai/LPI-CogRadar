const { Document, Packer, Paragraph, TextRun, HeadingLevel } = require('docx');
const fs = require('fs');

// 读取run_first_version_experiments.py的内容
const pythonCode = fs.readFileSync('run_first_version_experiments.py', 'utf-8');

// 创建文档
const doc = new Document({
  sections: [{
    children: [
      // 标题
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun('EP')],
      }),
      // 代码内容
      new Paragraph({
        children: [new TextRun(pythonCode)],
      }),
    ],
  }],
});

// 打包文档
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync('EP.docx', buffer);
  console.log('Word document created: EP.docx');
});
