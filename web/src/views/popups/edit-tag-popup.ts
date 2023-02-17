import m from 'mithril';
import { fetchRequest } from '../../utils/utils';
import { Popup, PopupAttrs } from './popup';
import { icons } from '../icons';
import { TagAttrs } from '../project-loader';

export interface EditTagPopupAttrs extends PopupAttrs {
  active: boolean;
  data: {
    tag: TagAttrs,
    projectId: string,
    uid: string,
  } | undefined;
}

export class EditTagPopup extends Popup implements m.ClassComponent<EditTagPopupAttrs> {
  private tagId: string | undefined;
  private uid: string | undefined;
  private projectId: string | undefined;

  constructor(vnode: m.CVnode<EditTagPopupAttrs>) {
    super();

    this.actions = [
      {name: 'Save', res: () => this.saveTag(vnode)},
      {name: 'Remove', res: () => this.removeTag(vnode)},
    ];
  }

  onupdate({attrs}: m.CVnodeDOM<EditTagPopupAttrs>): void {
    this.tagId = attrs.data?.tag._id;
    this.projectId = attrs.data?.projectId;
    this.uid = attrs.data?.uid;
  }

  saveTag(vnode: m.CVnode<EditTagPopupAttrs>): void {
    fetchRequest('/updateTag', {
      method: 'POST',
      body: {
        uid: this.uid,
        projectId: this.projectId,
        tagId: this.tagId,
        update: {
          name: this.input['tagName'],
          description: this.input['tagDescription'],
          goalQty: this.input['goalQty'],
        },
      }
    })
      .then(res => {
        this.hidePopup(vnode);
        vnode.attrs.reloadCallback();
      })
  }

  removeTag(vnode: m.CVnode<EditTagPopupAttrs>): void {
    fetchRequest('/removeTag', {
      method: 'POST',
      body: {
        uid: this.uid,
        projectId: this.projectId,
        tagId: this.tagId,
      }
    })
      .then(res => {
        this.hidePopup(vnode);
        vnode.attrs.reloadCallback();
      })
  }

  loadPopupContent({attrs}: m.CVnode<EditTagPopupAttrs>): m.Children {
    let tag = attrs.data?.tag;

    this.title = `Edit ${tag?.name}`;
    m.redraw();

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
        initialValue: tag?.goalQty.toString() || '',
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