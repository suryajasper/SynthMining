import React from 'react';
import { fetchRequest } from '../../utils/utils';
import { Popup, PopupAttrs } from './popup';
import { TagAttrs } from '../project-loader';

export interface EditTagPopupAttrs extends PopupAttrs {
  active: boolean;
  data: {
    tag: TagAttrs;
    projectId: string;
    uid: string;

    updateTag: (tagId: string, update: {
      name: string,
      description: string,
      goalQty: number,
    }) => void;
    removeTag: (tagId: string) => void;
  } | undefined;
}

export class EditTagPopup extends Popup {
  private tagId: string | undefined;
  private uid: string | undefined;
  private projectId: string | undefined;

  constructor(props: EditTagPopupAttrs) {
    super(props);

    this.setState({
      actions: [
        { name: 'Save', res: this.saveTag.bind(this) },
        { name: 'Remove', res: this.removeTag.bind(this) },
      ],
    });
  }

  componentDidUpdate(): void {
    this.tagId = this.props.data?.tag._id;
    this.projectId = this.props.data?.projectId;
    this.uid = this.props.data?.uid;
  }

  saveTag(): void {
    this.props.data.updateTag(this.tagId, {
      name: this.state.input['tagName'],
      description: this.state.input['tagDescription'],
      goalQty: parseInt(this.state.input['goalQty']),
    });
    this.hidePopup();
  }

  removeTag(): void {
    this.props.data.removeTag(this.tagId);
    this.hidePopup();
  }

  loadPopupContent(): JSX.Element[] {
    let tag = this.props.data?.tag;

    this.setState({ title: `Edit ${tag?.name}` });

    return [
      this.createInputGroup({
        id: 'tagName',
        displayTitle: 'Name',
        initialValue: tag?.name || '',
      }),
      this.createInputGroup({
        id: 'goalQty',
        displayTitle: 'Goal Quantity',
        type: 'number',
        initialValue: tag?.goalQty?.toString() || '',
      }),
      this.createInputGroup({
        id: 'tagDescription',
        displayTitle: 'Description',
        type: 'textarea',
        initialValue: tag?.description || '',
      }),
    ];
  }
}