import React, { Component } from "react";
import {Avatar, Button, Select} from "antd";
import ImageUnit from "./basasuya";

import "./imagelist.css";
import "./imagedecider.css";

const { Option } = Select;

class ImageDecider extends Component {
    constructor(props) {
        super(props)

        // this.state = {
        //     imagePath: ""
        // }
        this.knnOptions = [
          {
            "value": 3,
            "label": 3,
          },
          {
            "value": 4,
            "label": 4,
          },
          {
            "value": 5,
            "label": 5,
          },
          {
            "value": 6,
            "label": 6,
          },
          {
            "value": 7,
            "label": 7,
          },
          {
            "value": 8,
            "label": 8,
          },
          {
            "value": 9,
            "label": 9,
          },
          {
            "value": 10,
            "label": 10,
          },
        ];

        this.knnDefaultValue = 5;

        this.knnValue = 5;
    }

    handleClick() {
        if (this.props.onKnn) {
            this.props.onKnn(this.props.imageChoose.index, this.knnValue);
        }
    }

    handleSelect(k) {
        this.knnValue = k;

        // if (this.props.onKnn) {
        //     console.log("yingygingying", this.props.imageChoose.index)
        //     this.props.onKnn(this.props.imageChoose.index, k);
        // }
    }

    render() {
        const { knnOptions, knnDefaultValue } = this;

        return <div className="imagedecider">
            <ImageUnit path={this.props.imageChoose.path} />
            {/* <Avatar alt="C" src={this.props.path}  shape="square" size={100} /> */}
            <Button style={{
              marginLeft: 30,
              marginRight: 30
            }} icon="search" onClick={this.handleClick.bind(this)}>Search</Button>
            <Select defaultValue={knnDefaultValue} style={{ width: 96 }} onChange={this.handleSelect.bind(this)}>
              {knnOptions.map(option => <Option key={option.value} value={option.value}>{option.label}</Option>)}
            </Select>
        </div>
    }
}

export default ImageDecider