import cv2

#サンプル映像の読み込み
path = "こね.mov"
cap = cv2.VideoCapture(path)

print("動体検知を開始")

avg = None

while(True):
    #フレームを取得
    ret, frame = cap.read()
    
    if not ret:
        print("not capture")
        break
        
    #動体検知処理
    #グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 比較用にフレームの切り出し保存
    if avg is None:
        avg = gray.copy().astype("float")
        continue

    #現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    #画像を２値化する
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]

    #輪郭を抽出する
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 差分があった点を画面に描く
    for target in contours:
        x, y, w, h = cv2.boundingRect(target)

        if w < 30: continue #条件以下の変更点は除外

        #動体の位置を描画
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

    #画像表示
    cv2.imshow("Frame", frame)
    
    #キーボード入力処理
    key = cv2.waitKey(1)
    if key == 13: #enterキーの場合処理を抜ける
        break
        
#カメラデバイスクローズ
cap.release()

#ウィンドウクローズ
cv2.destroyAllWindows()
