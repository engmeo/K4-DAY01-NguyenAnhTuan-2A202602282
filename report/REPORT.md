# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 13/9/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.10 / PyTorch 2.x / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** ở muc ##2 phát hiện vật thể và ##3 phân đoạn đối tượng, chỗ nguồn evidence đã thay đổi từ sample `kitchen` thành sample `traffic`

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 :
  - `class_id`: 468
  - `class_name`: `cab`
  - `rank`: 1
  - `score`: 0.510915
  - `taxonomy_name`: `ImageNet-1K`
- Record này mô tả toàn ảnh như thế nào?
Record mô tả đối tượng/ngữ cảnh nổi bật nhất trong bức ảnh (cab). Do bài toán là Phân loại ảnh (Image Classification), mô hình gán 1 nhãn chung duy nhất cho toàn bộ khung hình mà không khoanh vùng vị trí từng đối tượng riêng biệt.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
Do người thiết kế/xây dựng bộ dữ liệu (Dataset Creators) quy định. Ở đây là tổ chức xây dựng bộ dữ liệu ImageNet-1K.
  - Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - ID: Tối ưu lưu trữ và xử lý máy tính.
  - Tên lớp: Trực quan cho con người đọc và kiểm tra.
  - Taxonomy: Định danh hệ quy chiếu ngữ nghĩa (tránh nhầm lẫn Class ID giữa các tập dữ liệu khác nhau như ImageNet vs COCO).
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
Guideline cần quy định ưu tiên gán nhãn cho đối tượng chính/diện tích lớn nhất, hoặc gán nhãn bối cảnh chung (như "traffic"). Nếu các chủ thể ngang nhau, cần quy định chuyển sang bài toán Multi-label hoặc Object Detection.
- Vì sao model score không phải ground truth?
Model score chỉ là xác suất tính toán dựa trên trọng số của thuật toán (có thể đưa ra dự đoán sai). Ground Truth phải do con người kiểm tra và xác nhận dựa trên guideline chuẩn. 	
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `traffic`.

- Một record mẫu:
  - `class_name`: `bus`
  - `score`: `0.912557` (91.26%)
  - `bbox_xyxy`: `[93.17, 187.95, 223.01, 320.91]`
  - `bbox_width`: `129.84` px
  - `bbox_height`: `132.96` px
- Diễn giải vị trí box bằng lời: Khung chữ nhật (bounding box) khoanh vùng chiếc xe bus này nằm ở phía bên trái bức ảnh, có góc trên-trái tại tọa độ pixel `(X1 = 93.17, Y1 = 187.95)` và góc dưới-phải tại tọa độ pixel `(X2 = 223.01, Y2 = 320.91)`. Khung có chiều rộng là 129.84 pixel và chiều cao là 132.96 pixel.
- So sánh số prediction ở hai threshold:
Ở threshold thấp(ví dụ: `0.25`): Mô hình trả về nhiều dự đoán hơn, giữ lại cả các vật thể xa, mờ hoặc bị che khuất một phần.
Ở threshold cao(ví dụ: `0.70`): Mô hình lọc bớt các dự đoán nghi ngờ, chỉ trả về ít box hơn tương ứng với các đối tượng mà mô hình cực kỳ chắc chắn.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
Threshold thấp: Giúp tăng độ bao phủ (Recall cao, tránh bỏ sót đối tượng) nhưng làm tăng khối lượng công việc cho Reviewer vì họ phải duyệt nhiều box nhiễu/dự đoán sai (False Positive).
Threshold cao: Giảm tải công việc cho Reviewer (chỉ duyệt ít box có độ tin cậy cao) nhưng nguy cơ bỏ sót vật thể thực tế lớn (Precision cao nhưng Recall thấp).
- Đề xuất một quy tắc box chặt:
Quy định cụ thể bằng con số pixel. Ví dụ: "Lề thừa từ điểm cực của vật thể đến mép box không được vượt quá 2–5 pixel".
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
Annotator tưởng tượng hình dáng trọn vẹn của vật thể và vẽ box bao trùm cả phần bị che.
## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `traffic`.

- Một record mẫu (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  - `instance_id`: `"traffic-001"`
  - `class_name`: `"bus"`
  - `score`: `0.925745` (92.57%)
  - **Số điểm polygon (`polygon_point_count`):** `120` điểm
  - **Một phần `polygon_xy` (10 điểm đầu tiên):** `[[148.0, 189.0], [147.0, 190.0], [145.0, 190.0], [143.0, 192.0], [142.0, 192.0], [141.0, 193.0], [139.0, 193.0], [137.0, 195.0], [136.0, 195.0], [135.0, 196.0]]`
- Polygon bổ sung chi tiết gì so với box?
Polygon mô tả chính xác đường ranh giới tự nhiên/hình dáng (contour) thực tế của vật thể tới từng điểm ảnh (pixel-level), thay vì chỉ ôm bằng một khung hình chữ nhật Bounding Box. Nó giúp bóc tách hoàn toàn vùng nền thừa ở các góc và xử lý tốt các vật thể có hình dạng bất quy tắc hoặc bị đứng nghiêng.
- `instance_id` dùng để làm gì và không phải loại ID nào?
`instance_id` (ví dụ: `"traffic-001"`) dùng để định danh duy nhất cho **một cá thể/vật thể cụ thể** trong bức ảnh, giúp phân biệt giữa chiếc xe bus này với chiếc xe bus khác (`"traffic-002"`). Đây **không phải** là `class_id` (mã danh mục lớp, ví dụ class_id `5` đại diện cho toàn bộ loài/danh mục "bus").
- Đề xuất một quy tắc biên mask:
Biên mask polygon phải bám sát đường viền tự nhiên của đối tượng với độ lệch không quá 2-3 pixel, không chờm ra nền xung quanh.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Vùng tiếp xúc đè lên nhau (Overlap): Guideline cần quy định rõ vật thể phía trước (Foreground) sẽ được vẽ mask đè lên vật thể phía sau (Background), tránh tạo ra vùng trống hoặc đè chung nhãn.
Vùng mờ/Bóng râm (Blur/Shadow): Guideline quy định chỉ gán nhãn phần thân vật thể thực sự, bỏ qua bóng râm đổ ra đường hoặc phần ảnh bị mờ chuyển tiếp.
Nếu vật thể bị cắt/che khuất thành 2 mảnh tách rời, guideline cần quy định vẽ 2 polygon độc lập nhưng gán chung 1 `instance_id`, hoặc escalate cho Lead khi không thể xác định 2 mảnh đó có thuộc cùng một đối tượng hay không.
## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | Nhãn danh mục tổng thể dạng văn bản (Single String Label, e.g., `"cab"`) | Ảnh chứa nhiều vật thể có độ lớn ngang nhau; khung cảnh phức tạp khó chọn 1 nhãn duy nhất. | Đọc Guideline, xác định đối tượng chính chiếm diện tích/tầm quan trọng lớn nhất và gán nhãn đại diện. | Kiểm tra xem nhãn được chọn có phản ánh đúng đối tượng chủ đạo hoặc ngữ cảnh toàn ảnh theo Guideline hay không. |
| **Phát hiện vật thể** | Bounding Box tọa độ chữ nhật (`[x1, y1, x2, y2, class_id]`) | Box thừa lề rộng; bỏ sót vật thể mờ/nhỏ; vẽ lẹm vào thân vật thể; không rõ quy tắc khi bị che khuất. | Khoanh box ôm sát ranh giới ngoài của vật thể, kiểm tra không bỏ sót đối tượng thuộc Taxonomy và áp dụng quy tắc che khuất. | Kiểm tra độ chặt của box (IoU), phát hiện các box vẽ thừa/thiếu, kiểm tra nhãn lớp và xác nhận các trường hợp che khuất/cắt mép. |
| **Instance segmentation** | Tập hợp các điểm Polygon/Mask ranh giới (`[[x1, y1], [x2, y2], ...]`) gắn kèm `instance_id` | Ranh giới bị mờ; các vật thể đè lên nhau (overlap) khó tách biệt; vẽ lem mask ra vùng nền xung quanh. | Chấm các điểm polygon bám sát đường viền tự nhiên của vật thể, tách riêng từng cá thể bằng `instance_id` khác nhau. | Soi chi tiết đường viền mask pixel-level, kiểm tra việc phân biệt đúng các `instance_id` riêng biệt và đảm bảo không lem ra nền. |

## 5. An toàn dữ liệu
 	
- Một quy tắc bảo vệ dữ liệu: Tuyệt đối không tự ý tải về, sao chép, lưu trữ cá nhân hoặc chia sẻ dữ liệu/hình ảnh của dự án ra bên ngoài môi trường làm việc được cấp phép, đồng thời không đưa thông tin định danh cá nhân (PII) vào mã nguồn hoặc báo cáo.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Quản lý dự án (Project Manager/Team Lead) hoặc Giảng viên/Trợ giảng phụ trách trực tiếp qua kênh liên lạc chính thức của khóa học.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
