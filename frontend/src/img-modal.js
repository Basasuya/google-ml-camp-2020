import React from "react";
import { Modal } from "antd";

import "./img-modal.css";

class ImgModal extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    const { enableModal, src, onCloseModal } = this.props;

    return (
      <div>
        <Modal
          visible={enableModal}
          centered={true}
          // closable={false}
          footer={null}
          width={700}
          height={700}
          onCancel={onCloseModal}
        >
          <img className="modal-img"
            src={src}
            // width={600}
            onClick={onCloseModal}
          />
        </Modal>
      </div>
    );
  }
}

export default ImgModal;
