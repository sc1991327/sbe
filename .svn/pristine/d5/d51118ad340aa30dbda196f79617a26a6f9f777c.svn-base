在parameter裡有一個epsilon_tex參數
是texture的閥值

在function detect()裡面
if(textdis(iter->aux.tex,frMagnitude))
{
	txResult->imageData[imagepixel] = 255;
}
else
{
	txResult->imageData[imagepixel] = 0;
}
這段是針對texture的結果做判斷

在fucntion textdis()裡面是比較model內和current frame的texture差值